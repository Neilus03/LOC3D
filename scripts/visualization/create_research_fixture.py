#!/usr/bin/env python3
"""Write clearly synthetic saved episode/proposal fixtures for CPU suite validation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import yaml

from delimit3d.visualization.contracts import Scene
from delimit3d.visualization.export_contracts import save_episode_export, save_proposal_export


def create(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    points, labels = [], []
    floor = rng.uniform([-1, -1, 0], [4, 3, .005], (700, 3))
    points.append(floor); labels.append(np.zeros(len(floor), dtype=np.int64))
    for index, center in enumerate(([0, 0, .7], [1.25, .05, .7], [2.6, 1.2, .45]), 1):
        samples = rng.uniform(-.5, .5, (650, 3))
        face = rng.integers(0, 3, len(samples))
        samples[np.arange(len(samples)), face] = rng.choice([-.5, .5], len(samples))
        points.append(samples + center); labels.append(np.full(len(samples), index, dtype=np.int64))
    xyz, truth = np.concatenate(points).astype(np.float32), np.concatenate(labels)
    palette = np.array([[.6, .6, .6], [.75, .45, .25], [.3, .55, .7], [.65, .7, .35]])
    scene = Scene("synthetic_contract_fixture", xyz, colors=palette[truth], labels=truth,
                  metadata={"synthetic": True, "dataset": "Synthetic fixture", "background_ids": [0]}).validate()
    scene_path = output / "scene.npz"
    np.savez_compressed(scene_path, points=xyz, colors=scene.colors, labels=truth)
    masks = np.zeros((6, len(xyz)), dtype=bool)
    masks[0] = (truth == 1) & (xyz[:, 0] <= 0)
    masks[1] = (truth == 1) & (xyz[:, 0] > 0)  # split
    masks[2] = truth == 2
    masks[3] = truth == 2                    # duplicate
    masks[4] = np.isin(truth, [1, 2])         # merge
    masks[5] = (truth == 0) & (xyz[:, 0] > 2) # false positive; object 3 missed
    proposal_path = save_proposal_export(output / "mask_only", scene,
        record={"synthetic": True, "output_contract": "mask_only", "prediction_domain": "raw_points",
                "mask_layout": "QN", "mask_values": "binary", "score_semantics": "handwritten synthetic scores",
                "evaluated_postprocessing": {"synthetic_fixture": True, "inference": False},
                "model": {}, "experiment_family": "synthetic_contract_validation"},
        arrays={"masks": masks, "query_ids": np.array([17, 23, 31, 45, 59, 71]),
                "scores": np.array([.65, .6, .9, .8, .55, .7])})
    first, second = int(np.flatnonzero(truth == 1)[0]), int(np.flatnonzero(truth == 2)[0])
    prediction0 = np.where(np.isin(truth, [1, 2]), truth, 0)
    prediction1 = np.where(truth == 1, 2, np.where(truth == 2, 1, 0))  # deliberate identity switch
    episode_path = save_episode_export(output / "episode", scene,
        record={"synthetic": True, "episode_id": "synthetic/identity_switch", "mode": "exploratory_replay",
            "targets": [{"target_id": "synthetic_object1", "label_id": 1}, {"target_id": "synthetic_object2", "label_id": 2}],
            "background_id": 0, "prediction_target_map": {"1": "synthetic_object1", "2": "synthetic_object2"},
            "prediction_domain": "raw_points", "metric_domain": "raw_points", "model": {},
            "click_budget": {"definition": "total synthetic click events", "maximum": 2,
                             "counts_background": True, "denominator": 1}, "click_policy": "handwritten synthetic fixture",
            "clicks": [{"click_id": f"synthetic-{order}", "order": order, "point_index": point,
                       "position": xyz[point].tolist(), "polarity": "positive", "target_id": f"synthetic_object{order+1}",
                       "assigned_label_id": order+1} for order, point in enumerate([first, second])],
            "frames": [{"frame_id": "before_identity_switch", "click_count": 1},
                       {"frame_id": "after_identity_switch", "click_count": 2}],
            "experiment_family": "synthetic_contract_validation"},
        arrays={"predictions": np.stack([prediction0, prediction1])})
    episode_name, proposal_name = "SYNTHETIC recorded episode", "SYNTHETIC mask-only proposals"
    config = {"version": 1, "seed": 42, "cpu_only": True, "max_display_points": 3000,
        "max_analysis_points": 3000, "visualizations": ["rgb", "labels", "prediction", "clicks", "confidence"],
        "scenes": [{"scene_id": scene.scene_id, "dataset": "Synthetic fixture", "kind": "npz", "path": str(scene_path),
            "metadata": scene.metadata, "queries": [],
            "models": [{"kind": "agile3d_episode", "name": episode_name, "path": str(episode_path)},
                       {"kind": "mask3d_proposals", "name": proposal_name, "path": str(proposal_path)}],
            "recipes": [
                {"name": "synthetic_temporal_identity", "columns": 4,
                 "panels": ["rgb", f"{episode_name}/episode_target", f"{episode_name}/episode_step_0", f"{episode_name}/episode_step_1"],
                 "caption": "SYNTHETIC FIXTURE. Handwritten masks deliberately exchange target identities after the second click. "
                            "Stable colors expose the regression. No trained model, inference, or experimental evidence."},
                {"name": "synthetic_mask_only_contract", "columns": 4,
                 "panels": ["rgb", f"{proposal_name}/proposal_gt", f"{proposal_name}/proposal_display", f"{proposal_name}/proposal_errors"],
                 "caption": "SYNTHETIC FIXTURE. Handwritten overlapping proposals exercise splits, merges, duplicates, false positives "
                            "and a missed object. Display assignment is separate from independent proposal diagnostics; no official AP."},
                {"name": "synthetic_proposal_inspection", "columns": 3,
                 "panels": [f"{proposal_name}/proposal_{i}" for i in [2, 3, 5]],
                 "caption": "SYNTHETIC FIXTURE. Separate duplicate and false-positive proposals retain original query identities and "
                            "handwritten scores. These are adapter tests, not research results."}]}]}
    config_path = output / "synthetic.validation.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(create(args.output.resolve()))


if __name__ == "__main__":
    main()
