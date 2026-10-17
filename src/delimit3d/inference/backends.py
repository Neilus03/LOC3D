"""Real, strict-checkpoint inference for the released LitePT task decoders.

Only administrator-configured full downstream checkpoints are loaded here.
No optimizer, original encoder checkpoint, or decoder initialization is needed.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from delimit3d.runtime.config import expand_environment, load_config, merge_config

if TYPE_CHECKING:
    from .scene import Scene


def _scene_fingerprint(scene: Scene) -> str:
    digest = hashlib.sha256()
    for value in (scene.points, scene.colors, scene.normals, scene.segments):
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def prepare_features(scene: Scene, *, channels: int, color_mean=None, color_std=None) -> np.ndarray:
    """RGB uint8 -> unit RGB; preserve supplied normals in their original frame."""
    if channels not in (3, 6):
        raise ValueError("Inference currently supports RGB3 and RGBN6 LitePT inputs")
    rgb = np.asarray(scene.colors, dtype=np.float32) / 255.0
    if (color_mean is None) != (color_std is None):
        raise ValueError("color_mean and color_std must be supplied together")
    if color_mean is not None:
        mean, std = np.asarray(color_mean), np.asarray(color_std)
        if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid RGB normalization configuration")
        rgb = (rgb - mean) / std
    if channels == 6:
        if scene.normals is None:
            raise ValueError("This RGBN6 model requires surface normals. Load a mesh with normals or an NPZ containing normals.")
        rgb = np.concatenate((rgb, scene.normals), axis=1)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def quantize_interactive(points: np.ndarray, grid_size: float):
    """Official AGILE3D minimum shift and lexicographic first representatives."""
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    shifted = np.asarray(points, dtype=np.float32) - np.min(points, axis=0)
    # Official compatibility quantization divides by float64 voxel sizes.
    grid = np.floor(shifted / np.full(3, grid_size, dtype=np.float64)).astype(np.int32)
    unique, representatives, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    return shifted[representatives], unique, representatives, inverse


def prepare_clicks(clicks: list[dict], inverse: np.ndarray, max_events: int):
    """Map original point indices and arbitrary user object IDs into decoder IDs."""
    if len(clicks) > max_events:
        raise ValueError(f"This model supports at most {max_events} click events")
    normalized = []
    for click in clicks:
        index, object_id = click.get("point_index"), click.get("object_id", 1)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(inverse):
            raise ValueError("Each click needs a valid original point_index")
        if isinstance(object_id, bool) or not isinstance(object_id, (int, np.integer)) or object_id < 0:
            raise ValueError("Click object_id must be a non-negative integer")
        if click.get("positive", True) is False:
            object_id = 0
        normalized.append((int(inverse[index]), int(object_id)))
    object_ids = sorted({object_id for _, object_id in normalized if object_id > 0})
    if not object_ids:
        raise ValueError("Add at least one foreground click before interactive inference")
    remap = {original: position + 1 for position, original in enumerate(object_ids)}
    remap[0] = 0
    indices, times = {i: [] for i in remap.values()}, {i: [] for i in remap.values()}
    for time, (index, object_id) in enumerate(normalized):
        indices[remap[object_id]].append(index)
        times[remap[object_id]].append(time)
    return indices, times, np.asarray(object_ids, dtype=np.int32)


def constrain_token_labels(labels: np.ndarray, clicks: list[dict], inverse: np.ndarray, object_ids: np.ndarray) -> np.ndarray:
    """Honor click constraints at voxel resolution; the latest click wins conflicts."""
    result = np.array(labels, copy=True)
    remap = {int(value): index + 1 for index, value in enumerate(object_ids)}
    remap[0] = 0
    for click in clicks:
        object_id = 0 if click.get("positive", True) is False else int(click.get("object_id", 1))
        result[int(inverse[click["point_index"]])] = remap[object_id]
    return result


def _checkpoint_state(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Expected a full downstream checkpoint mapping")
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(payload.get(key), dict):
            state = payload[key]
            break
    else:
        raise ValueError("Full task checkpoint must contain model_state_dict, model, or state_dict; encoder-only weights cannot run segmentation")
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    if not any(key.startswith("backbone.") for key in state) or not any(key.startswith("decoder.") for key in state):
        raise ValueError("Full task checkpoint must contain both backbone and trained decoder weights")
    return state


class ModelRunner:
    """One loaded model. Serialize calls; the web service owns request locking."""

    def __init__(self, entry: dict, device: str = "cuda"):
        import torch
        from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder
        from delimit3d.models.litept_wrapper import LitePTBackbone
        from delimit3d.tasks.mask3d.model import LitePTMask3DModel

        self.torch = torch
        self.entry = dict(entry)
        self.task = entry["task"]
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Start inference inside a GPU allocation using the documented LitePT environment.")
        if self.device.type != "cuda":
            raise ValueError("LitePT inference requires a CUDA device; CPU is supported for input validation and UI tests only")
        from delimit3d.evaluation.features import configure_attention_backend
        self.attention_backend = configure_attention_backend(self.device, "auto")
        self.config = expand_environment(merge_config(
            load_config(entry["config"], expand_env=False), entry.get("config_overrides", {})), strict=False)
        self.config["task"] = self.task
        self.seed = int(self.config.get("seed", self.config.get("experiment", {}).get("seed", 42)))
        settings = self.config.get("model", {})
        variant = settings.get("litept_variant", "litept_s_star")
        backbone_name = settings.get("backbone_name", "litept_s_star")
        if variant != "litept_s_star" or (isinstance(backbone_name, str) and backbone_name not in ("litept", "litept_s_star")):
            raise ValueError("The inference adapter currently supports LitePT-S* task checkpoints")
        root = settings.get("litept_root") or self.config.get("paths", {}).get("litept_root")
        if not root or not Path(root).expanduser().is_dir():
            raise ValueError("Set model.litept_root (Mask3D) or paths.litept_root (AGILE3D) to the prepared LitePT checkout")
        root = str(Path(root).expanduser().resolve())
        self.channels = int(settings.get("input_channels", settings.get("in_channels", 6)))
        self.grid_size = float(settings.get("voxel_size", self.config.get("protocol", {}).get("voxel_size_m", 0.05)))
        if not np.isfinite(self.grid_size) or self.grid_size <= 0:
            raise ValueError("Voxel size must be finite and positive")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            if self.task == "mask3d":
                if int(settings.get("num_classes", 2)) != 2 or settings.get("mask_only", False):
                    raise ValueError("The automatic adapter requires a trained class-agnostic objectness head (num_classes=2)")
                self.model = LitePTMask3DModel(
                    litept_root=root, backbone_frozen=False,
                    backbone_kwargs={"in_channels": self.channels, "grid_size": self.grid_size,
                        "litept_variant": variant, "multi_scale": False,
                        "feature_pyramid_mode": settings.get("feature_pyramid_mode", "mask3d_fpn"),
                        "voxel_reduce": settings.get("voxel_reduce", "representative"),
                        "representative_sampling": settings.get("representative_sampling", "first"),
                        "cache_training_voxelization": False},
                    decoder_kwargs={"train_queries": int(settings.get("train_queries", 100)),
                        "eval_queries": int(settings.get("eval_queries", 100)), "num_classes": 2,
                        "mask_only": False})
                self.bn_policy = self.config.get("eval", {}).get("backbone_batchnorm_policy", "train_no_update")
            elif self.task == "agile3d":
                # These are the same modules and state keys as Agile3DModel. Loading
                # the entire fine-tuned state makes original initialization files unnecessary.
                self.model = torch.nn.Module()
                self.model.backbone = LitePTBackbone(litept_root=root, in_channels=self.channels,
                    grid_size=self.grid_size, litept_variant=variant, multi_scale=False,
                    cache_training_voxelization=False, feature_pyramid_mode=None,
                    voxel_reduce="representative", representative_sampling="first")
                keys = ("feature_dim", "hidden_dim", "num_heads", "dim_feedforward", "num_decoders",
                        "num_bg_queries", "dropout", "pre_norm", "max_click_events", "normalize_pos_enc", "gauss_scale", "aux")
                self.model.decoder = Agile3DClickDecoder(**{key: settings[key] for key in keys if key in settings})
                self.bn_policy = self.config.get("evaluation", {}).get("backbone_batchnorm_policy", "train_no_update")
            else:
                raise ValueError(f"Unsupported inference task: {self.task}")
        if self.bn_policy not in ("train_no_update", "eval"):
            raise ValueError("backbone_batchnorm_policy must be train_no_update or eval")
        checkpoint = Path(entry["checkpoint"]).expanduser().resolve(strict=True)
        # Checkpoints are trusted administrator inputs, never user-uploaded files.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model_contract") is not None:
            from delimit3d.tasks.contracts import model_contract
            if payload["model_contract"] != model_contract(self.config):
                raise ValueError("Checkpoint model/readout contract differs from the selected inference configuration")
        self.model.load_state_dict(_checkpoint_state(payload), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        pointrope_modules = {type(module).__module__ for module in self.model.backbone.modules()
                             if type(module).__name__ == "PointROPE"}
        self.pointrope_backend = "torch" if any(name.endswith("pointrope_torch") for name in pointrope_modules) else "cuda"
        from delimit3d.runtime.records import file_sha256
        self.provenance = {"model_id": entry["id"], "task": self.task,
            "attention_backend": self.attention_backend,
            "pointrope_backend": self.pointrope_backend,
            "checkpoint_sha256": file_sha256(checkpoint),
            "resolved_config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "inference_seed": self.seed, "serialization_policy": "seeded_per_scene_torch_rng_restored",
            "model_contract_checked": payload.get("model_contract") is not None,
            "checkpoint": str(checkpoint), "epoch": int(payload.get("epoch", -1)),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "completed_epochs": int(payload.get("epoch", -1)) + (1 if self.task == "agile3d" else 0),
            "checkpoint_epoch_convention": "zero_based_epoch_index" if self.task == "agile3d" else "completed_epoch_count",
            "voxel_size_m": self.grid_size, "input_channels": self.channels,
            "backbone_batchnorm_policy": self.bn_policy, "checkpoint_loading": "strict_full_task",
            "semantic_labels": False}
        self._scene_key = None
        self._encoded = None
        self._automatic = None

    @contextmanager
    def _inference_rng(self):
        """Repeat serialization orders for every scene encode without leaking RNG state."""
        torch = self.torch
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            torch.random.default_generator.manual_seed(self.seed)
            with torch.cuda.device(index):
                torch.cuda.manual_seed(self.seed)
            yield

    def _normalization(self):
        from delimit3d.evaluation.backbone_batchnorm import current_batch_without_updates
        return current_batch_without_updates(self.model.backbone) if self.bn_policy == "train_no_update" else nullcontext()

    def predict(self, scene: Scene, clicks: list[dict], score_threshold: float = 0.5) -> dict:
        if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        key = _scene_fingerprint(scene)
        if key != self._scene_key:
            self._encoded = None
            self._automatic = None
            self._scene_key = key
        if self.task == "mask3d":
            if clicks:
                raise ValueError("Automatic Mask3D does not accept clicks; choose an interactive model")
            return self._predict_automatic(scene, float(score_threshold))
        return self._predict_interactive(scene, clicks)

    def _tensor(self, array, *, integer=False):
        torch = self.torch
        return torch.as_tensor(np.ascontiguousarray(array), device=self.device,
                               dtype=torch.long if integer else torch.float32)

    def _predict_automatic(self, scene: Scene, threshold: float) -> dict:
        torch = self.torch
        if self._automatic is None:
            data = self.config.get("data", {})
            features = prepare_features(scene, channels=self.channels,
                color_mean=data.get("color_mean"), color_std=data.get("color_std"))
            if scene.segments is None:
                grid = np.floor(scene.points / self.grid_size).astype(np.int64)
                _, segments = np.unique(grid, axis=0, return_inverse=True)
                segment_source = "voxel_fallback_not_benchmark_segments"
            else:
                _, segments = np.unique(scene.segments, return_inverse=True)
                segment_source = "provided_segments"
            with torch.inference_mode(), self._inference_rng(), self._normalization():
                result = self.model(self._tensor(scene.points), self._tensor(features),
                                    self._tensor(segments, integer=True))[0]
                logits = result["mask_logits"]
                probabilities = logits.sigmoid()
                masks = logits > 0
                # Raw-query objectness x mean foreground mask probability,
                # matching the class-agnostic evaluation confidence convention.
                quality = (probabilities * masks).sum(1) / masks.sum(1).clamp_min(1)
                scores = (result["object_logits"].softmax(-1)[:, 0] * quality).cpu().numpy()
                self._automatic = (masks.cpu().numpy(), scores, segment_source)
        masks, scores, segment_source = self._automatic
        selected = np.flatnonzero((scores >= threshold) & masks.any(axis=1))
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        labels = np.zeros(len(scene.points), dtype=np.int32)
        # Display an exclusive assignment; exported instance IDs refer to the
        # retained raw queries. Higher-confidence overlapping masks win.
        instance_ids = np.arange(1, len(selected) + 1, dtype=np.int32)
        for label, query in zip(instance_ids, selected):
            labels[masks[query] & (labels == 0)] = label
        return {"labels": labels, "scores": np.asarray(scores[selected], dtype=np.float32),
            "instance_ids": instance_ids,
            "packed_masks": np.packbits(masks[selected].T, axis=0),
            "mask_point_count": np.asarray(len(scene.points), dtype=np.int64),
            "provenance": {**self.provenance, "segments": segment_source,
                "score_threshold": threshold, "score": "objectness_times_mean_mask_probability",
                "packed_masks_bitorder": "big", "display_assignment": "highest_confidence_mask_wins", "raw_query_indices": selected.tolist()}}

    def _predict_interactive(self, scene: Scene, clicks: list[dict]) -> dict:
        torch = self.torch
        if self._encoded is None:
            features = prepare_features(scene, channels=self.channels)
            xyz, grid, representatives, inverse = quantize_interactive(scene.points, self.grid_size)
            with torch.inference_mode(), self._inference_rng(), self._normalization():
                output = self.model.backbone(self._tensor(xyz), self._tensor(features[representatives]),
                    point_offsets=self._tensor(np.asarray([len(xyz)]), integer=True),
                    grid_coord=self._tensor(grid, integer=True))
            self._encoded = (output.scene_tokens, output.scene_xyz, inverse)
        tokens, xyz, inverse = self._encoded
        indices, times, object_ids = prepare_clicks(clicks, inverse, self.model.decoder.max_click_events)
        with torch.inference_mode():
            result = self.model.decoder(tokens, xyz, clicks=indices, click_times=times)
            probabilities = result["pred_masks"].softmax(-1).cpu().numpy()
        token_labels = constrain_token_labels(probabilities.argmax(-1), clicks, inverse, object_ids)
        scores = np.asarray([float(probabilities[token_labels == i, i].mean())
            if np.any(token_labels == i) else 0.0 for i in range(1, len(object_ids) + 1)], dtype=np.float32)
        compact_labels = token_labels[inverse]
        lookup = np.concatenate((np.asarray([0], dtype=np.int32), object_ids))
        labels = lookup[compact_labels]
        return {"labels": labels, "scores": scores, "instance_ids": object_ids,
            "provenance": {**self.provenance, "click_count": len(clicks),
                "coordinate_contract": "official_agile3d_per_axis_minimum",
                "background_object_id": 0, "click_constraints": "latest_click_wins_at_voxel_resolution", "point_mapping": "first_voxel_representative_inverse",
                "score": "mean_assigned_object_probability"}}
