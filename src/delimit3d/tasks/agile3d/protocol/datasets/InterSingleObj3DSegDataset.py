# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data

from torch.utils.data import Dataset
from delimit3d.tasks.agile3d.protocol.quantization import ME

import numpy as np
import os
import json

from delimit3d.tasks.agile3d.protocol.utils.ply import read_ply
from delimit3d.tasks.agile3d.protocol.datasets.normal_utils import load_scene_normals
from delimit3d.tasks.agile3d.protocol.datasets.external_feature_utils import load_scene_features

from copy import deepcopy

class InterSingleObj3DSegDataset(Dataset):
    def __init__(self, scan_folder, object_list, quantization_size,
                 crop=False, transforms=None, use_normals=False, normal_root=None,
                 external_feature_root=None, external_feature_dim=None):
        super(InterSingleObj3DSegDataset, self).__init__()

        self.quantization_size = quantization_size
        self.scan_folder = scan_folder
        self.use_normals = use_normals
        self.normal_root = normal_root
        self.external_feature_root = external_feature_root
        self.external_feature_dim = external_feature_dim

        if self.external_feature_root and self.crop:
            raise ValueError(
                'External frozen features currently require full-scene mode; '
                'crop features are not aligned to the official ScanNet cache.'
            )
        if self.external_feature_root and transforms:
            raise ValueError(
                'External frozen features are aligned to the unaugmented '
                'official ScanNet geometry; disable geometric transforms.'
            )


        self.dataset_list = np.load(object_list)

        self.dataset_size = len(self.dataset_list)

        self.crop = crop
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, i):

        scene_name = self.dataset_list[i,0]
        object_id = self.dataset_list[i,1]

        if self.crop:
            point_cloud = read_ply(os.path.join(self.scan_folder, scene_name, scene_name + '_crop_' + object_id + '.ply'))
            labels_full = point_cloud['label'].astype(np.int32)
            if self.use_normals and not self.external_feature_root:
                raise ValueError(
                    'RGB+normal crop mode requires crop-aligned normal sidecars; '
                    'the released ScanNet bundle does not provide them'
                )

        else:
            point_cloud = read_ply(os.path.join(self.scan_folder, scene_name + '.ply'))
            labels_full = (point_cloud['label'] == int(object_id)).astype(np.int32)

        coords_x = point_cloud['x'] - point_cloud['x'].min()
        coords_y = point_cloud['y'] - point_cloud['y'].min()
        coords_z = point_cloud['z'] - point_cloud['z'].min()
        coords_full = np.column_stack([coords_x, coords_y, coords_z]).astype(np.float32)
        colors_full = None
        if not self.external_feature_root:
            colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']]).astype(np.float32) / 255.0
        normals_full = None
        if self.use_normals and not self.external_feature_root:
            normals_full = load_scene_normals(
                self.normal_root, scene_name, coords_full.shape[0]
            )

        if self.transforms:
            coords_full, normals_full = self.augment(coords_full, normals_full)

        if normals_full is not None:
            colors_full = np.concatenate([colors_full, normals_full], axis=1)

        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords_full,
            quantization_size=self.quantization_size,
            return_index=True,
            return_inverse=True)

        raw_coords_qv = coords_full[unique_map]
        if self.external_feature_root:
            feats_qv = load_scene_features(
                self.external_feature_root,
                scene_name,
                coords_qv,
                expected_dim=self.external_feature_dim,
            )
        else:
            feats_qv = colors_full[unique_map]
        labels_qv = labels_full[unique_map]

        click_idx_qv = {}

        return coords_qv, raw_coords_qv, feats_qv, labels_qv, labels_full, inverse_map, click_idx_qv, scene_name, object_id


    def augment(self, point_cloud, normals=None):
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            if normals is not None:
                normals[:, 0] = -1 * normals[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            if normals is not None:
                normals[:, 1] = -1 * normals[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle_pre = np.random.choice([0, np.pi/2, np.pi, np.pi/2*3])
        rot_mat_pre = self.rotz(rot_angle_pre)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat_pre))
        if normals is not None:
            normals[:, 0:3] = np.dot(normals[:, 0:3], np.transpose(rot_mat_pre))

        rot_angle = (np.random.random() * 2* np.pi) - np.pi  # -180 ~ +180 degree
        rot_mat = self.rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        if normals is not None:
            normals[:, 0:3] = np.dot(normals[:, 0:3], np.transpose(rot_mat))

        return point_cloud, normals

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def collation_fn(data_labels):
    coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, object_id = list(zip(*data_labels)) 
    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = [torch.from_numpy(l) for l in labels]
    raw_coords_batch = torch.from_numpy(np.concatenate(raw_coords, 0)).float()
    labels_full = [torch.from_numpy(l) for l in labels_full]

    return coords_batch, raw_coords_batch, feats_batch, labels_batch, labels_full, inverse_map, click_idx, scene_name, object_id

def make_scan_transforms(split, external_features=False):
    if external_features:
        return False

    if split=='train':
        return True
    else:
        return False

    raise ValueError(f'unknown {split}')

def build(split, args):

    PATHS = {
        "train": (args.scan_folder, args.train_list),
        "val": (args.scan_folder, args.val_list)
    }

    scan_folder, object_list = PATHS[split]
    
    dataset = InterSingleObj3DSegDataset(
        scan_folder,
        object_list,
        args.voxel_size,
        crop=args.crop,
        transforms=make_scan_transforms(
            split, external_features=bool(getattr(args, 'external_feature_root', None))
        ),
        use_normals=getattr(args, 'use_normals', False),
        normal_root=getattr(args, 'normal_root', None),
        external_feature_root=getattr(args, 'external_feature_root', None),
        external_feature_dim=getattr(args, 'external_feature_dim', None),
    )
    
    return dataset, collation_fn
