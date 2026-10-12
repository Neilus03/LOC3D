"""Pretraining gradients extracted from the executed trainer."""

from __future__ import annotations


import math


from typing import Any, Mapping, Sequence

import numpy as np

import torch

import torch.distributed as dist


def _global_auxiliary_budget_passed(controller: Mapping[str, Any], *, auxiliary_enabled: bool) -> bool:
    """An absent auxiliary objective is not a failed measured gradient budget."""
    if controller.get("rejected", False):
        return False
    budget = controller.get("auxiliary_budget", {})
    if auxiliary_enabled:
        return budget.get("passed") is True
    # Require the exact inactive state rather than ignoring a budget failure
    # or any accidentally enabled auxiliary objective.
    return (controller.get("objectives") == {} and budget.get("passed") is None
            and budget.get("violation_count") == 0
            and budget.get("last_global_sum_aux_norm_ratio") is None)


def _gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def _clip_gradient_diagnostics(
    parameters: Any,
    *,
    max_norm: float,
) -> dict[str, float]:
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    parameters = list(parameters)
    pre_clip_norm = _gradient_norm(parameters)
    clip_return_norm = float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    )
    post_clip_norm = _gradient_norm(parameters)
    clip_coefficient = min(1.0, max_norm / (clip_return_norm + 1.0e-6))
    return {
        "pre_clip_norm": pre_clip_norm,
        "post_clip_norm": post_clip_norm,
        "clip_return_norm": clip_return_norm,
        "clip_coefficient": clip_coefficient,
        "was_clipped": float(clip_return_norm > max_norm),
    }


def _telemetry_objective_cosine(
    records: Sequence[Mapping[str, Any]], objective: str
) -> float | None:
    values: list[float] = []
    for record in records:
        stages = (
            record.get("objectives", {})
            .get(objective, {})
            .get("stages", {})
        )
        for metrics in stages.values():
            if metrics.get("cosine_valid"):
                value = float(metrics["cosine_vs_final_dec0"])
                if math.isfinite(value):
                    values.append(value)
    return float(np.mean(values)) if values else None


def _telemetry_auxiliary_budget(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_record: list[float] = []
    per_objective: dict[str, list[float]] = {}
    for record in records:
        objectives = record.get("objectives", {})
        stage_sums: dict[str, float] = {}
        for objective, objective_record in objectives.items():
            if objective == "final_dec0":
                continue
            ratios = [
                float(metrics["grad_norm_ratio_vs_final_dec0"])
                for metrics in objective_record.get("stages", {}).values()
                if metrics.get("norm_ratio_valid")
            ]
            if ratios:
                value = max(ratios)
                per_objective.setdefault(str(objective), []).append(value)
                stage_sums[str(objective)] = value
        if stage_sums:
            per_record.append(sum(stage_sums.values()))
    maximum = max(per_record) if per_record else None
    return {
        "conservative_sum_aux_norm_ratio_max": maximum,
        "conservative_sum_aux_norm_ratio_mean": (
            float(np.mean(per_record)) if per_record else None
        ),
        "per_objective_max_ratio": {
            objective: max(values)
            for objective, values in per_objective.items()
        },
        "limit": 0.5,
        "passed": maximum is not None and maximum <= 0.5,
    }


def _manual_objective_gradients(
    *,
    components: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    scale: float,
) -> dict[str, list[torch.Tensor]]:
    """Materialize one microbatch's objective gradients for rare PCGrad fallback."""

    if scale <= 0.0:
        raise ValueError("manual gradient scale must be positive")
    output: dict[str, list[torch.Tensor]] = {}
    for objective, component in components.items():
        gradients = torch.autograd.grad(
            component / float(scale),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        output[objective] = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if gradient is None
            else gradient.detach().clone()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    return output


def _flat_objective_gradients(
    *,
    components: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    scale: float,
) -> dict[str, torch.Tensor]:
    """Return compact FP32 objective gradients over one shared parameter set.

    These vectors are used only on predeclared telemetry panels or after the
    conditional PCGrad fallback has been enabled.  Keeping one flat vector per
    objective avoids retaining full-model zero tensors for private heads and
    permits one collective per objective instead of one per parameter.
    """

    if scale <= 0.0:
        raise ValueError("flat gradient scale must be positive")
    if not parameters:
        raise ValueError("shared parameter set must not be empty")
    parameter_numels = [int(parameter.numel()) for parameter in parameters]
    total_numel = sum(parameter_numels)
    output: dict[str, torch.Tensor] = {}
    for objective, component in components.items():
        gradients = torch.autograd.grad(
            component / float(scale),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        flat = torch.empty(
            total_numel,
            dtype=torch.float32,
            device=parameters[0].device,
        )
        offset = 0
        for parameter, gradient, numel in zip(
            parameters, gradients, parameter_numels, strict=True
        ):
            target = flat[offset : offset + numel]
            if gradient is None:
                target.zero_()
            else:
                target.copy_(gradient.detach().reshape(-1).float())
            offset += numel
        output[str(objective)] = flat
    return output


def _add_flat_objective_gradients(
    destination: dict[str, torch.Tensor] | None,
    source: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if destination is None:
        return {objective: vector.clone() for objective, vector in source.items()}
    if set(destination) != set(source):
        raise RuntimeError("Flat objective set drift inside a logical batch")
    for objective, vector in source.items():
        if destination[objective].shape != vector.shape:
            raise RuntimeError("Flat objective shape drift inside a logical batch")
        destination[objective].add_(vector)
    return destination


def _synchronize_accumulated_gradients(
    *,
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
) -> None:
    """Average gradients accumulated inside a ``DistributedDataParallel.no_sync`` block.

    DDP normally performs this reduction from the next synchronized backward.
    A V2 logical batch can skip its nominal final scene before constructing a
    DDP graph, though, so there is no backward pass available to flush the
    preceding ``no_sync`` gradients.  Reduce a shared presence mask first so
    ranks with different local autograd usage still execute identical
    collectives, and preserve ``None`` for parameters unused on every rank.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    trainable = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    if not trainable or world_size == 1:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Accumulated-gradient synchronization requires an initialized process group"
        )
    device = trainable[0].device
    if any(parameter.device != device for parameter in trainable):
        raise ValueError("Accumulated-gradient parameters must share one device")
    present = torch.tensor(
        [int(parameter.grad is not None) for parameter in trainable],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    sparse = torch.tensor(
        int(
            any(
                parameter.grad is not None and parameter.grad.is_sparse
                for parameter in trainable
            )
        ),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(sparse, op=dist.ReduceOp.MAX)
    if int(sparse.item()) != 0:
        raise RuntimeError(
            "Accumulated-gradient synchronization does not support sparse gradients"
        )
    for parameter, globally_present in zip(trainable, present.tolist(), strict=True):
        if int(globally_present) == 0:
            parameter.grad = None
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
            parameter.grad = gradient
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(float(world_size))


def _global_shared_gradient_report(
    *,
    accumulated: Mapping[str, torch.Tensor],
    shared_parameters: Sequence[torch.nn.Parameter],
    projected_objectives: set[str],
    world_size: int,
    install_projected_gradient: bool,
) -> dict[str, Any]:
    """Reduce true logical-batch gradients and optionally install shared PCGrad.

    Private head gradients are deliberately left untouched.  Only the common
    LitePT backbone is eligible for projection, which prevents an auxiliary
    objective from injecting the primary objective into private parameters.
    """

    if "final_dec0" not in accumulated:
        raise ValueError("Shared-gradient report requires final_dec0")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    reduced = {
        objective: vector.detach().clone().float()
        for objective, vector in accumulated.items()
    }
    expected_numel = sum(int(parameter.numel()) for parameter in shared_parameters)
    if not shared_parameters or any(
        int(vector.numel()) != expected_numel for vector in reduced.values()
    ):
        raise ValueError("Shared-gradient vector/parameter layout mismatch")
    for vector in reduced.values():
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        vector.div_(float(world_size))

    reference = reduced["final_dec0"]
    reference_square = reference.square().sum()
    projected_vectors = {
        objective: vector.clone() for objective, vector in reduced.items()
    }
    objective_reports: dict[str, Any] = {}
    for objective, vector in reduced.items():
        if objective == "final_dec0":
            continue
        objective_square = vector.square().sum()
        dot_before = (vector * reference).sum()
        denominator_before = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * objective_square.clamp_min(1.0e-20)
        )
        should_project = (
            objective in projected_objectives
            and float(dot_before.detach()) < 0.0
            and float(reference_square.detach()) > 0.0
        )
        projected = projected_vectors[objective]
        if should_project:
            coefficient = dot_before / reference_square.clamp_min(1.0e-20)
            projected.sub_(coefficient * reference)
        projected_square = projected.square().sum()
        dot_after = (projected * reference).sum()
        denominator_after = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * projected_square.clamp_min(1.0e-20)
        )
        objective_reports[objective] = {
            "enabled": objective in projected_objectives,
            "applied": bool(should_project),
            "global_shared_dot_before": float(dot_before.detach()),
            "global_shared_cosine_before": float(
                (dot_before / denominator_before).detach()
            ),
            "global_shared_norm_ratio_before": float(
                torch.sqrt(
                    objective_square.clamp_min(0.0)
                    / reference_square.clamp_min(1.0e-20)
                ).detach()
            ),
            "global_shared_dot_after": float(dot_after.detach()),
            "global_shared_cosine_after": float(
                (dot_after / denominator_after).detach()
            ),
            "global_shared_norm_ratio_after": float(
                torch.sqrt(
                    projected_square.clamp_min(0.0)
                    / reference_square.clamp_min(1.0e-20)
                ).detach()
            ),
        }

    auxiliary_names = [
        objective for objective in reduced if objective != "final_dec0"
    ]
    auxiliary_before = sum(
        (reduced[objective] for objective in auxiliary_names),
        torch.zeros_like(reference),
    )
    auxiliary_after = sum(
        (projected_vectors[objective] for objective in auxiliary_names),
        torch.zeros_like(reference),
    )
    reference_norm = reference.norm().clamp_min(1.0e-20)
    budget_before = float((auxiliary_before.norm() / reference_norm).detach())
    budget_after = float((auxiliary_after.norm() / reference_norm).detach())

    if install_projected_gradient:
        combined = sum(
            projected_vectors.values(), torch.zeros_like(reference)
        )
        offset = 0
        for parameter in shared_parameters:
            numel = int(parameter.numel())
            replacement = combined[offset : offset + numel].reshape_as(parameter)
            parameter.grad = replacement.to(dtype=parameter.dtype).clone()
            offset += numel
        if offset != int(combined.numel()):
            raise RuntimeError("Shared-gradient installation layout drift")

    return {
        "schema_version": "litept_global_shared_backbone_gradient/v2",
        "reference": "final_dec0",
        "shared_parameter_count": len(shared_parameters),
        "shared_parameter_numel": expected_numel,
        "projected_objectives": sorted(projected_objectives),
        "projection_installed": bool(install_projected_gradient),
        "objectives": objective_reports,
        "auxiliary_budget": {
            "global_sum_aux_norm_ratio_before": budget_before,
            "global_sum_aux_norm_ratio_after": budget_after,
            "limit": 0.5,
            "passed_before": budget_before <= 0.5,
            "passed_after": budget_after <= 0.5,
        },
    }


def _install_global_flat_total_gradient(
    *,
    accumulated: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
) -> None:
    """All-reduce and install one compact total-gradient vector."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    expected_numel = sum(int(parameter.numel()) for parameter in parameters)
    if int(accumulated.numel()) != expected_numel:
        raise ValueError("Flat total-gradient vector/parameter layout mismatch")
    reduced = accumulated.detach().clone().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced.div_(float(world_size))
    offset = 0
    for parameter in parameters:
        numel = int(parameter.numel())
        parameter.grad = (
            reduced[offset : offset + numel]
            .reshape_as(parameter)
            .to(dtype=parameter.dtype)
            .clone()
        )
        offset += numel


def _add_manual_gradients(
    destination: dict[str, list[torch.Tensor]] | None,
    source: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    if destination is None:
        return {
            objective: [tensor.clone() for tensor in tensors]
            for objective, tensors in source.items()
        }
    if set(destination) != set(source):
        raise RuntimeError("Manual objective set drift inside a logical batch")
    for objective, tensors in source.items():
        for accumulated, current in zip(
            destination[objective], tensors, strict=True
        ):
            accumulated.add_(current)
    return destination


def _install_global_pcgrad(
    *,
    accumulated: Mapping[str, Sequence[torch.Tensor]],
    parameters: Sequence[torch.nn.Parameter],
    projected_objectives: set[str],
    world_size: int,
    shared_parameter_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Compatibility PCGrad helper with projection restricted to shared tensors."""

    if "final_dec0" not in accumulated:
        raise ValueError("PCGrad requires final_dec0 as reference")
    reduced = {
        objective: [tensor.clone() for tensor in tensors]
        for objective, tensors in accumulated.items()
    }
    for tensors in reduced.values():
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(float(world_size))
    reference = reduced["final_dec0"]
    if shared_parameter_indices is None:
        shared_parameter_indices = set(range(len(parameters)))
    if not shared_parameter_indices or any(
        index < 0 or index >= len(parameters)
        for index in shared_parameter_indices
    ):
        raise ValueError("shared_parameter_indices are empty or out of range")
    projection_report: dict[str, Any] = {}
    for objective, gradients in reduced.items():
        if objective == "final_dec0":
            continue
        dot = sum(
            (gradient.float() * ref.float()).sum()
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            )
            if index in shared_parameter_indices
        )
        reference_square = sum(
            ref.float().square().sum()
            for index, ref in enumerate(reference)
            if index in shared_parameter_indices
        )
        objective_square = sum(
            gradient.float().square().sum()
            for index, gradient in enumerate(gradients)
            if index in shared_parameter_indices
        )
        projected = objective in projected_objectives and float(dot) < 0.0
        if projected and float(reference_square) > 0.0:
            coefficient = dot / reference_square.clamp_min(1.0e-20)
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            ):
                if index in shared_parameter_indices:
                    gradient.sub_(coefficient.to(gradient.dtype) * ref)
        dot_after = sum(
            (gradient.float() * ref.float()).sum()
            for index, (gradient, ref) in enumerate(
                zip(gradients, reference, strict=True)
            )
            if index in shared_parameter_indices
        )
        objective_square_after = sum(
            gradient.float().square().sum()
            for index, gradient in enumerate(gradients)
            if index in shared_parameter_indices
        )
        denominator = torch.sqrt(
            reference_square.clamp_min(1.0e-20)
            * objective_square.clamp_min(1.0e-20)
        )
        projection_report[objective] = {
            "enabled": objective in projected_objectives,
            "applied": bool(projected),
            "global_dot_before": float(dot),
            "global_cosine_before": float(dot / denominator),
            "global_shared_dot_after": float(dot_after),
            "global_shared_cosine_after": float(
                dot_after
                / torch.sqrt(
                    reference_square.clamp_min(1.0e-20)
                    * objective_square_after.clamp_min(1.0e-20)
                )
            ),
        }
    for parameter_index, parameter in enumerate(parameters):
        combined = sum(
            (
                tensors[parameter_index]
                for tensors in reduced.values()
            ),
            torch.zeros_like(parameter, memory_format=torch.preserve_format),
        )
        parameter.grad = combined
    return {
        "schema_version": "litept_global_shared_parameter_pcgrad/v2",
        "reference": "final_dec0",
        "shared_parameter_indices": sorted(shared_parameter_indices),
        "projected_objectives": sorted(projected_objectives),
        "objectives": projection_report,
    }
