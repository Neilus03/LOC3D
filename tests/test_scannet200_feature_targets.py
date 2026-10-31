"""Target identity and class-eligibility checks for the ScanNet200 diagnostic."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement


from delimit3d.evaluation import scannet200_feature_targets as targets


class ScanNet200FeatureTargetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "processed" / "val" / "scene0000_00"
        self.source.mkdir(parents=True)
        # Raw instance 0 is a chair; small ceiling remains an instance. Wall,
        # floor, ignored vertices, and an unknown-category object stay input.
        self.instance = np.array([0] * 101 + [1, 2, 3, 4, -1], np.int64)
        self.semantic = np.array([1] * 101 + [0, 2, 35, -1, -1], np.int64)
        self.xyz = np.column_stack((np.arange(106) * .017,
                                    np.arange(106) % 3 * .02,
                                    np.arange(106) % 5 * .01)).astype(np.float32)
        for name, array in {"coord": self.xyz,
                            "color": np.full((106, 3), 127, np.uint8),
                            "normal": np.tile([0., 0., 2.], (106, 1)).astype(np.float32),
                            "instance": self.instance, "segment200": self.semantic}.items():
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

    def test_instance_zero_ceiling_and_full_point_support(self):
        record = self.prepare()
        arrays, objects = targets.load_scene(record)
        self.assertEqual([obj["gt_instance_id"] for obj in objects], [0, 3])
        self.assertEqual([obj["class_name"] for obj in objects], ["chair", "ceiling"])
        self.assertEqual([obj["class_index"] for obj in objects], [1, 35])
        self.assertEqual([obj["official_min_region_100_eligible"] for obj in objects], [True, False])
        self.assertEqual(record["eligibility"]["official_min_region_100_objects"], 1)
        self.assertEqual(record["background_points"], 4)
        self.assertTrue(np.array_equal(arrays["xyz"], self.xyz))
        self.assertEqual(len(arrays["point_to_cell"]), 106)
        self.assertTrue(np.all(arrays["labels"][:101] == 0))
        self.assertEqual(arrays["labels"][103], 1)
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
        np.save(self.source / "segment200.npy", self.semantic)
        with self.assertRaisesRegex(ValueError, "mixes semantic classes"):
            self.prepare()

    def test_reordered_raw_mesh_is_rejected(self):
        self._mesh(self.xyz[::-1])
        with self.assertRaisesRegex(ValueError, "original mesh vertex order"):
            self.prepare()

    def test_sealed_source_replacement_is_rejected(self):
        self.prepare()
        np.save(self.source / "color.npy", np.zeros((106, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "different sealed targets"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
