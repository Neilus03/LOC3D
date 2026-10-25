"""Sanitized ScanNet LA data: geometry plus fixed observed training labels."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.aligned_augmentation import augment_points_mask3d_scannet_rgbn6_aligned
from .data.single_scene_dataset import build_input_features
from .sparse_supervision import SparseAnnotations, VALID_NYU40, THING_NYU40


class LowAnnotationDataset(Dataset):
    """Training never opens dense annotations, packs, or validation archives."""

    def __init__(self, root, scene_ids, *, train, budget, seed=42, granularity="g05"):
        self.root = Path(root)
        self.ids = list(scene_ids)
        self.train = bool(train)
        self.budget = int(budget)
        self.seed = int(seed)
        self.granularity = granularity
        self.epoch = 1
        if self.budget not in (20, 50, 100, 200):
            raise ValueError("Use an official 20/50/100/200 annotation budget")
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("The scene split must be nonempty and unique")
        self._scene_dirs = [self.root / "geometry" / f"{s}.npz" for s in self.ids]

    def __len__(self):
        return len(self.ids)

    @property
    def scene_ids(self):
        return list(self.ids)

    def __getitem__(self, index):
        padding = False
        if isinstance(index, tuple):
            index, padding = index
        scene = self.ids[index]
        with np.load(self._scene_dirs[index], allow_pickle=False) as archive:
            if set(archive.files) != {"points", "colors", "normals"}:
                raise ValueError("LA geometry may contain only points, colors, normals")
            points, colors, normals = [archive[k].copy() for k in ("points", "colors", "normals")]
        n = len(points)
        if n == 0 or any(value.shape != (n, 3) or not np.isfinite(value).all()
                         for value in (points, colors, normals)):
            raise ValueError(f"Invalid geometry or aligned features: {scene}")
        meta = {"limited_annotation": True}
        if self.train:
            path = self.root / "train_annotations" / str(self.budget) / f"{scene}.npz"
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"vertex_indices", "instance_ids", "semantic_ids"}:
                    raise ValueError("Unexpected fields in sanitized LA annotations")
                indices, identities, semantics = [archive[k].copy() for k in
                                                   ("vertex_indices", "instance_ids", "semantic_ids")]
            if any(value.shape != (self.budget,) or value.dtype.kind not in "iu"
                   for value in (indices, identities, semantics)):
                raise ValueError("Fixed official annotation entries must be aligned integer arrays")
            if np.any(indices < 0) or np.any(indices >= n):
                raise ValueError("Official annotation index is outside the original mesh")
            valid = np.isin(semantics, list(VALID_NYU40))
            if (np.any(identities < -1) or np.any((identities == -1) != (semantics == -1))
                    or np.any(valid != (identities >= 0))
                    or np.any((identities > 0) != np.isin(semantics, list(THING_NYU40)))):
                raise ValueError("Sanitized annotation identity/semantic states disagree")
            # Duplicated official entries are preserved, and cannot disagree.
            for vertex in np.unique(indices):
                at = indices == vertex
                if len(np.unique(np.column_stack((identities[at], semantics[at])), axis=0)) != 1:
                    raise ValueError("Duplicate official vertex has conflicting labels")
            augmentation_seed = int.from_bytes(hashlib.sha256(
                f"{self.seed}/{self.epoch}/{scene}".encode()).digest()[:4], "little")
            np_state, py_state, torch_state = np.random.get_state(), random.getstate(), torch.random.get_rng_state()
            try:
                np.random.seed(augmentation_seed)
                random.seed(augmentation_seed)
                torch.random.default_generator.manual_seed(augmentation_seed)
                points, colors, normals = augment_points_mask3d_scannet_rgbn6_aligned(
                    points, colors, normals=normals)
            finally:
                np.random.set_state(np_state)
                random.setstate(py_state)
                torch.random.set_rng_state(torch_state)
            if len(points) != n:
                raise ValueError("LA forbids cropping or dropping original vertices")
            labels = np.full(n, -1, dtype=np.int64)
            labels[indices] = identities
            supervised = np.zeros(n, dtype=bool)
            supervised[indices] = True
            meta.update(la_indices=indices.tolist(), la_semantics=semantics.tolist(),
                        augmentation_seed=augmentation_seed)
            classes = None
        else:
            with np.load(self.root / "validation" / f"{scene}.npz", allow_pickle=False) as archive:
                if set(archive.files) != {"instance_ids", "instance_class_pairs"}:
                    raise ValueError("Unexpected fields in dense validation annotations")
                labels = archive["instance_ids"].copy()
                pairs = archive["instance_class_pairs"].copy()
            if labels.shape != (n,) or labels.dtype.kind not in "iu" or pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("Validation annotation domain differs from original geometry")
            classes = {int(k): int(v) for k, v in pairs}
            supervised = np.ones(n, dtype=bool)
        features = build_input_features(points, colors, use_colors=True, append_xyz=False,
                                        use_normals=True, normals=normals,
                                        color_mean=[0., 0., 0.], color_std=[1., 1., 1.])
        points = torch.from_numpy(points).float()
        # The engine derives the exact geometric voxel inverse on its model
        # device. Annotation-derived mesh segments never enter this loader.
        segments = torch.zeros(n, dtype=torch.long)
        return dict(scene_id=scene, scene_dir=str(self._scene_dirs[index]), points=points,
                    features=torch.from_numpy(features).float(),
                    labels_by_granularity={self.granularity: torch.from_numpy(labels).long()},
                    instance_classes_by_granularity={self.granularity: classes},
                    valid_points=torch.ones(n, dtype=torch.bool), seen_points=torch.ones(n, dtype=torch.bool),
                    supervision_mask=torch.from_numpy(supervised), granularities=(self.granularity,),
                    scene_meta=meta, scan_segments=segments, original_num_points=n, eval_input_points=n,
                    full_scene=True, subsampling_mode="none", is_padding=padding)

    def get_full_item(self, index):
        return self[index]


def make_sparse_targets(batch, granularity, device, *, class_aware):
    if class_aware:
        raise ValueError("The released LA protocol uses class-agnostic instance targets")
    targets = []
    for labels, meta in zip(batch.labels_by_granularity[granularity], batch.scene_meta, strict=True):
        if not meta.get("limited_annotation") or "la_indices" not in meta:
            raise ValueError("Dense validation labels reached LA training target construction")
        indices = torch.tensor(meta["la_indices"], dtype=torch.long)
        targets.append(SparseAnnotations(indices, labels[indices],
                                         torch.tensor(meta["la_semantics"], dtype=torch.long),
                                         len(labels)).to(device))
    return targets
