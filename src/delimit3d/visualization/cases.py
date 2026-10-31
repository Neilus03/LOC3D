"""Versioned, deterministic case selection with explicit candidate populations."""
from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA = "delimit3d_case_collection/v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def select_cases(candidates: list[dict], *, seed: int = 42, representative_count: int = 2,
                 effect_count: int = 1, tie_tolerance: float = 0.01) -> dict:
    """Select within dataset/experiment strata; effect tags never imply prevalence.

    Required keys: case_id, scene_id, dataset, experiment. Optional comparison:
    reference_value/model_value, metric, metric_domain, budget_definition and
    quantitative_rows. Scores must be comparable before classifying effects.
    """
    if representative_count < 0 or effect_count < 0 or tie_tolerance < 0:
        raise ValueError("Selection counts and tolerance must be nonnegative")
    by_id = {}
    for raw in candidates:
        row = dict(raw)
        for key in ("case_id", "scene_id", "dataset", "experiment"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Each candidate requires nonempty {key}")
        if row["case_id"] in by_id:
            raise ValueError("Candidate case_id values must be unique")
        comparison = row.get("comparison")
        if comparison is not None:
            for key in ("reference_value", "model_value", "metric", "metric_domain", "budget_definition", "unit"):
                if key not in comparison:
                    raise ValueError(f"Comparison requires {key}")
            import math
            if not all(math.isfinite(comparison[k]) for k in ("reference_value", "model_value")):
                raise ValueError("Comparison values must be finite")
            if comparison.get("higher_is_better", True) is not True:
                raise ValueError("Normalize lower-is-better comparisons explicitly before selection")
        by_id[row["case_id"]] = row
    population = [by_id[k] for k in sorted(by_id)]
    strata = {}
    for row in population:
        strata.setdefault((row["dataset"], row["experiment"]), []).append(row)
    selected = {}
    def add(row, tag, reason):
        item = selected.setdefault(row["case_id"], {**row, "selection_tags": [], "selection_reasons": []})
        item["selection_tags"].append(tag)
        item["selection_reasons"].append(reason)
    for stratum, rows in sorted(strata.items()):
        ranking = sorted(rows, key=lambda row: canonical_hash([seed, row["case_id"]]))
        for row in ranking[:representative_count]:
            add(row, "representative", "Lowest SHA256(seed, case_id) within dataset/experiment; outcome independent")
        comparable = [r for r in rows if r.get("comparison")]
        # Metric semantics define separate effect populations, even within one experiment.
        groups = {}
        for row in comparable:
            c = row["comparison"]
            groups.setdefault(tuple(c[k] for k in ("metric", "metric_domain", "budget_definition", "unit")), []).append(row)
        for same_metric in groups.values():
            tolerance = float(same_metric[0]["comparison"].get("tie_tolerance", tie_tolerance))
            if tolerance < 0 or any(float(r["comparison"].get("tie_tolerance", tie_tolerance)) != tolerance for r in same_metric):
                raise ValueError("Each metric/unit stratum needs one nonnegative tie tolerance")
            delta = lambda r: r["comparison"]["model_value"] - r["comparison"]["reference_value"]
            for tag, pool, key in (
                ("win", [r for r in same_metric if delta(r) > tolerance], lambda r: (-delta(r), r["case_id"])),
                ("regression", [r for r in same_metric if delta(r) < -tolerance], lambda r: (delta(r), r["case_id"])),
                ("tie", [r for r in same_metric if abs(delta(r)) <= tolerance], lambda r: (abs(delta(r)), r["case_id"])),
                ("difficult", same_metric, lambda r: (r["comparison"]["model_value"], r["case_id"])),
            ):
                for row in sorted(pool, key=key)[:effect_count]:
                    add(row, tag, f"Effect-selected {tag}; tolerance {tolerance:g} in recorded metric units")
    return {"schema": SCHEMA, "version": 1, "seed": seed,
            "population_sha256": canonical_hash(population), "candidate_count": len(population),
            "candidate_population": population,
            "method": {"representative": "SHA256(seed, case_id) ranked within dataset/experiment",
                       "representative_count_per_stratum": representative_count,
                       "effect_count_per_metric_stratum": effect_count, "tie_tolerance": tie_tolerance,
                       "effect_selection_is_population_estimate": False},
            "cases": [selected[k] for k in sorted(selected)]}


def validate_collection(collection: dict) -> dict:
    if collection.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}")
    population = collection["candidate_population"]
    if canonical_hash(sorted(population, key=lambda x: x["case_id"])) != collection["population_sha256"]:
        raise ValueError("Candidate population hash mismatch")
    if len({r["case_id"] for r in population}) != len(population):
        raise ValueError("Duplicate candidate identities")
    ids = {r["case_id"] for r in population}
    if any(r["case_id"] not in ids for r in collection["cases"]):
        raise ValueError("Selected case is outside candidate population")
    if collection.get("candidate_count") != len(population):
        raise ValueError("Candidate count mismatch")
    by_id = {r["case_id"]: r for r in population}
    selected_ids = [r["case_id"] for r in collection["cases"]]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate selected case identities")
    for row in collection["cases"]:
        source_fields = {k:v for k,v in row.items() if k not in {"selection_tags", "selection_reasons"}}
        if source_fields != by_id[row["case_id"]]:
            raise ValueError("Selected case differs from candidate record")
    method=collection.get("method",{})
    expected=select_cases(population, seed=collection["seed"],
        representative_count=method["representative_count_per_stratum"],
        effect_count=method["effect_count_per_metric_stratum"], tie_tolerance=method["tie_tolerance"])
    if collection["cases"] != expected["cases"]:
        raise ValueError("Selected cases differ from recorded deterministic selection rule")
    return collection


def resolve_scene_cases(config: dict, base_dir) -> tuple[list[dict], dict | None]:
    """Bind scene manifests to validated collection rows, retaining source identities."""
    from copy import deepcopy
    from pathlib import Path
    from .adapters import resolve_path, sha256_file
    scenes = deepcopy(config['scenes'])
    collection = None
    if config.get('case_collection'):
        path = resolve_path(config['case_collection'], base_dir)
        collection = validate_collection(json.loads(path.read_text()))
        collection = {**collection, 'source_path':str(path), 'source_sha256':sha256_file(path)}
        population = {r['case_id']:r for r in collection['candidate_population']}
        chosen = {r['case_id']:r for r in collection['cases']}
    for spec in scenes:
        metadata = spec.get('metadata',{})
        spec.setdefault('case_id',metadata.get('case_id',spec.get('scene_id')))
        spec.setdefault('experiment',metadata.get('experiment',metadata.get('experiment_family',config.get('experiment','unspecified'))))
        spec.setdefault('selection',metadata.get('selection',{}))
        spec.setdefault('quantitative_rows',metadata.get('quantitative_rows',[]))
        if collection is not None:
            case_id=spec['case_id']
            if case_id not in population:raise ValueError(f'Scene case_id is outside case collection: {case_id}')
            row=population[case_id]
            for key in ('scene_id','dataset'):
                if spec.get(key)!=row[key]:raise ValueError(f'Case {case_id} {key} differs from candidate record')
            if 'query_original_point_id' in row and spec.get('queries')!=[row['query_original_point_id']]:
                raise ValueError(f'Case {case_id} queries differ from original candidate identity')
            spec['experiment']=row['experiment']
            selected=chosen.get(case_id)
            spec['selection_tags']=selected['selection_tags'] if selected else ['available_unselected']
            spec['selection']={'case_id':case_id,'population_sha256':collection['population_sha256'],
                'candidate_count':collection['candidate_count'],'method':collection['method'],
                'tags':spec['selection_tags'],'reasons':selected['selection_reasons'] if selected else
                ['Additional available artifact, outside the selected representative/effect subset'],
                'candidate_record':row,'collection_source_sha256':collection['source_sha256']}
            spec['quantitative_rows']=row.get('quantitative_rows',[])
        # Hash all linked row sources; an absent metric file remains explicitly unavailable.
        refs=spec.get('quantitative_rows',[])
        refs=[refs] if isinstance(refs,dict) else refs
        def enrich(value):
            if isinstance(value,dict):
                result={k:enrich(v) for k,v in value.items()}
                p=value.get('path')
                if isinstance(p,str) and (p.startswith('/') or Path(p).suffix in {'.json','.jsonl','.csv'}):
                    source=resolve_path(p,base_dir)
                    result['source_available']=source.is_file()
                    if source.is_file():
                        digest=sha256_file(source)
                        if value.get('sha256') and value['sha256']!=digest:
                            raise ValueError(f'Quantitative source SHA mismatch: {source}')
                        result['sha256']=digest
                return result
            if isinstance(value,list):return [enrich(v) for v in value]
            return value
        spec['quantitative_rows']=enrich(refs)
    return scenes,collection
