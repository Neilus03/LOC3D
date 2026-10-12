"""Verify portable code/input records before any production optimizer step."""
import json
import os
from pathlib import Path
import time
from delimit3d.runtime.records import file_sha256, write_json
from delimit3d.data.partfield_contrastive import negative_policy_name

SCHEMA = "delimit3d_pretraining_execution/v1"


def verify_code_record(path):
    record = json.loads(Path(path).read_text())
    if record.get("schema_version") != "delimit3d_code_manifest/v1" or record.get("passed") is not True:
        raise ValueError("Expected a passed release code record")
    package = Path(__file__).resolve().parents[1]
    checkout = Path(record["snapshot_root"])
    if not record.get("files"):
        raise ValueError("Code record is empty")
    for name, expected in record["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid code record path")
        source = (package.joinpath(*relative.parts[2:]) if relative.parts[:2] == ("src", "delimit3d")
                  else checkout / relative)
        if file_sha256(source) != expected:
            raise ValueError(f"Executed code differs from the saved code record: {name}")
    required = {"src/delimit3d/" + str(source.relative_to(package))
                for source in package.rglob("*")
                if source.is_file() and "__pycache__" not in source.parts
                and source.suffix in {".py", ".json", ".patch", ".list"}}
    if missing := required.difference(record["files"]):
        raise ValueError(f"Code record omits executed package files: {sorted(missing)}")
    return record


def input_bindings(args):
    verify_code_record(args.code_manifest)
    audit = json.loads(Path(args.source_audit).read_text())
    if audit.get("passed") is not True or audit.get("artifact_hashes_verified") is not True:
        raise ValueError("Pretraining requires a passed artifact-verified source audit")
    return {key + "_sha256": file_sha256(getattr(args, key)) for key in
            ("code_manifest", "source_audit", "initial_checkpoint", "train_split", "holdout_split")}


def validate_execution_record(payload, args):
    if payload.get("schema_version") != SCHEMA or payload.get("passed") is not True:
        raise ValueError("Execution record did not pass")
    if payload.get("trust_root") != input_bindings(args):
        raise ValueError("Execution record inputs or code changed")
    flag = payload.get("allow_shared_proposal_negatives", False)
    policy = negative_policy_name(flag)
    if flag != getattr(args, "allow_shared_proposal_negatives", False) or payload.get("negative_policy", policy) != policy:
        raise ValueError("Execution record negative policy differs from the recipe")
    return payload


def prepare_execution_record(args):
    if args.execution_provenance is not None or args.smoke_manifest is not None or args.smoke_manifest_list is not None:
        return
    path = args.output_dir.with_name(args.output_dir.name + "_execution.json")
    expected = {"schema_version": SCHEMA, "passed": True, "trust_root": input_bindings(args)}
    flag = getattr(args, "allow_shared_proposal_negatives", False)
    if getattr(args, "source_granularity", None) == "sam2":
        expected.update(allow_shared_proposal_negatives=flag, negative_policy=negative_policy_name(flag))
    if int(os.environ.get("RANK", "0")) == 0:
        if path.exists() and json.loads(path.read_text()) != expected:
            raise ValueError("Existing execution record differs; choose a fresh run directory")
        if not path.exists():
            write_json(path, expected)
    deadline = time.monotonic() + 600
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Rank zero did not publish the execution record")
        time.sleep(.2)
    validate_execution_record(json.loads(path.read_text()), args)
    args.execution_provenance = path
    args.execution_provenance_sha256 = file_sha256(path)
