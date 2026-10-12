"""Fail-closed, CPU-only historical LitePT initialization audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import torch

from delimit3d.evaluation.agile3d_decoder import state_sha256

SOURCE_SHA256 = "8dcfc95c7826d5591890d146e01279260113d3af23e33a6586151a02fdbc3499"
SOURCE_BYTES = 192211847


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_state(model, source):
    """Only mapping: raw LitePT key k -> wrapper key 'model.' + k."""
    expected = model.state_dict()
    mapped = {}
    report = {"mapping_rule": "target = 'model.' + source; no reshape or dtype conversion",
              "source_tensor_count": len(source), "target_tensor_count": len(expected),
              "additional_source_tensors": [], "missing_target_tensors": [],
              "incompatible_tensors": [], "nonfinite_source_tensors": [],
              "tensor_mapping": [], "batchnorm": []}
    for key, value in source.items():
        if not torch.is_tensor(value):
            report["incompatible_tensors"].append({"source": key, "reason": "not a tensor"})
            continue
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            report["nonfinite_source_tensors"].append(key)
        target = "model." + key
        if target not in expected:
            report["additional_source_tensors"].append(key)
            continue
        row = {"source": key, "target": target, "shape": list(value.shape),
               "dtype": str(value.dtype), "expected_shape": list(expected[target].shape),
               "expected_dtype": str(expected[target].dtype)}
        report["tensor_mapping"].append(row)
        mapped[target] = value
        if value.shape != expected[target].shape or value.dtype != expected[target].dtype:
            report["incompatible_tensors"].append(row)
    report["missing_target_tensors"] = sorted(set(expected) - set(mapped))
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        keys = [name + "." + k for k in module.state_dict()]
        valid = all(k in mapped and mapped[k].shape == expected[k].shape
                    and mapped[k].dtype == expected[k].dtype for k in keys)
        if valid:
            valid = (all(not mapped[k].is_floating_point() or bool(torch.isfinite(mapped[k]).all())
                         for k in keys)
                     and bool((mapped[name + ".running_var"] >= 0).all())
                     and int(mapped[name + ".num_batches_tracked"]) >= 0)
        report["batchnorm"].append({"module": name, "keys": keys, "passed": valid,
            "num_batches_tracked": int(mapped[name + ".num_batches_tracked"])
            if name + ".num_batches_tracked" in mapped else None})
    report["passed"] = not any(report[k] for k in (
        "missing_target_tensors", "incompatible_tensors", "nonfinite_source_tensors"))
    report["passed"] &= all(x["passed"] for x in report["batchnorm"])
    return report, mapped


def load_historical(model, checkpoint, report_path=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    digest = file_hash(checkpoint)
    if digest != SOURCE_SHA256 or checkpoint.stat().st_size != SOURCE_BYTES:
        raise ValueError("Historical checkpoint file identity mismatch")
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError("Historical checkpoint archive integrity failed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (set(payload) != {"epoch", "step", "model", "criterion", "optimizer", "scheduler", "args"}
            or payload["epoch"] != 11 or payload["step"] != 215776):
        raise ValueError("Historical payload identity mismatch")
    source = payload["model"]
    report, mapped = audit_state(model, source)
    report.update(schema="historical_julius_litept_compatibility/v1", path=str(checkpoint),
                  file_sha256=digest, bytes=checkpoint.stat().st_size, payload_epoch=11,
                  payload_step=215776, root_entries=list(payload), archive_integrity=True,
                  load_policy="torch.load(map_location='cpu', weights_only=True); model only",
                  restored_old_training_state=False)
    report["passed"] &= len(source) == len(model.state_dict()) == 302
    if report["passed"]:
        model.load_state_dict(mapped, strict=True)
        if not all(torch.equal(model.state_dict()[k], v) for k, v in mapped.items()):
            raise RuntimeError("Loaded tensors differ from the audited source")
        report["state_sha256"] = state_sha256(model.state_dict())
        report["loaded_tensors"] = len(mapped)
        report["all_loaded_tensors_equal_source"] = True
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Report content is stable except for the staged path; retain each invocation.
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise ValueError("Strict historical compatibility failed: " + json.dumps({
            k: v for k, v in report.items() if k != "tensor_mapping"}))
    return report
