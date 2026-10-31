"""Target identity and class-eligibility checks for the ScanNet20 diagnostic."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement


from delimit3d.evaluation import scannet20_feature_targets as targets


class ScanNet20FeatureTargetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "processed" / "val" / "scene0000_00"
        self.source.mkdir(parents=True)
        # Raw instance 0 is a chair; wall, floor, and otherfurniture are
        # retained as distinct instances. Unknown and unannotated vertices
        # remain in the input and count as leakage.
        self.instance = np.array([0] * 101 + [1, 2, 3, 4, -1], np.int64)
        self.semantic = np.array([4] * 101 + [0, 1, 19, -1, -1], np.int64)
        self.xyz = np.column_stack((np.arange(106) * .017,
                                    np.arange(106) % 3 * .02,
                                    np.arange(106) % 5 * .01)).astype(np.float32)
        for name, array in {"coord": self.xyz,
                            "color": np.full((106, 3), 127, np.uint8),
                            "normal": np.tile([0., 0., 2.], (106, 1)).astype(np.float32),
                            "instance": self.instance, "segment20": self.semantic}.items():
            np.save(self.source / f"{name}.npy", array)
        self.mesh = self.root / "scene0000_00_vh_clean_2.ply"
        self._mesh(self.xyz)

    def tearDown(self):
        self.temporary.cleanup()

    def _mesh(self, xyz):
        vertices = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, i]
        PlyData([PlyElement.describe(vertices, "vertex")]).write(self.mesh)

    def prepare(self, **kwargs):
        return targets.prepare_scene("scene0000_00", self.source, self.root / "targets",
                                     mesh_path=self.mesh, **kwargs)

    def test_instance_zero_wall_floor_and_full_point_support(self):
        record = self.prepare()
        arrays, objects = targets.load_scene(record)
        self.assertEqual([obj["gt_instance_id"] for obj in objects], [0, 1, 2, 3])
        self.assertEqual([obj["class_name"] for obj in objects], ["chair", "wall", "floor", "otherfurniture"])
        self.assertEqual([obj["class_index"] for obj in objects], [4, 0, 1, 19])
        self.assertEqual([obj["official_min_region_100_eligible"] for obj in objects], [True, False, False, False])
        self.assertEqual(record["eligibility"]["official_min_region_100_objects"], 1)
        self.assertEqual(record["background_points"], 2)
        self.assertTrue(np.array_equal(arrays["xyz"], self.xyz))
        self.assertEqual(len(arrays["point_to_cell"]), 106)
        self.assertTrue(np.all(arrays["labels"][:101] == 0))
        self.assertEqual(arrays["labels"][103], 3)
        self.assertTrue(np.allclose(np.linalg.norm(arrays["normal"], axis=1), 1))
        self.assertEqual(objects[0]["query_vertex"], int(np.square(self.xyz[:101] - self.xyz[:101].mean(axis=0)).sum(axis=1).argmin()))
        self.assertEqual(self.prepare(), record)

    def test_minimum_size_filters_target_but_keeps_vertices(self):
        record = self.prepare(min_instance_points=100)
        arrays, objects = targets.load_scene(record)
        self.assertEqual(len(objects), 1)
        self.assertEqual(len(arrays["xyz"]), 106)
        self.assertEqual(arrays["labels"][103], -1)
        self.assertEqual(record["background_points"], 5)

    def test_mixed_semantic_instance_is_rejected(self):
        self.semantic[20] = 3
        np.save(self.source / "segment20.npy", self.semantic)
        with self.assertRaisesRegex(ValueError, "mixes semantic classes"):
            self.prepare()

    def test_reordered_raw_mesh_is_rejected(self):
        self._mesh(self.xyz[::-1])
        with self.assertRaisesRegex(ValueError, "original mesh vertex order"):
            self.prepare()

    def test_mixed_ignored_and_valid_semantics_are_rejected(self):
        self.semantic[20] = -1
        np.save(self.source / "segment20.npy", self.semantic)
        with self.assertRaisesRegex(ValueError, "mixes semantic classes"):
            self.prepare()

    def test_semantically_equal_raw_instances_remain_distinct(self):
        self.semantic[102] = 0
        np.save(self.source / "segment20.npy", self.semantic)
        record = self.prepare()
        self.assertEqual([obj["class_name"] for obj in record["objects"]],
                         ["chair", "wall", "wall", "otherfurniture"])
        self.assertEqual([obj["raw_instance_id"] for obj in record["objects"]], [0, 1, 2, 3])

    def test_parent_geometry_reused_exactly(self):
        parent = self.prepare()
        record = targets.prepare_scene("scene0000_00", self.source, self.root / "from_parent",
                                       mesh_path=self.mesh, parent_scene=parent)
        before, _ = targets.load_scene(parent)
        after, _ = targets.load_scene(record)
        for key in ("xyz", "rgb255", "normal", "shift", "common_grid", "point_to_cell"):
            self.assertTrue(np.array_equal(before[key], after[key]), key)
        self.assertTrue(record["geometry_parent"]["geometry_and_common_grid_reused_exactly"])

    def test_parent_source_replacement_is_rejected(self):
        parent = self.prepare()
        np.save(self.source / "color.npy", np.zeros((106, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "parent color source identity mismatch"):
            targets.prepare_scene("scene0000_00", self.source, self.root / "from_parent",
                                  mesh_path=self.mesh, parent_scene=parent)

    def test_sealed_source_replacement_is_rejected(self):
        self.prepare()
        np.save(self.source / "color.npy", np.zeros((106, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "different sealed targets"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
