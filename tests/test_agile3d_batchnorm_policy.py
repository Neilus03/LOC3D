"""Protect eval/training separation and exact equivalence to the Mask3D policy."""
import copy
import unittest

import torch
from torch import nn

from delimit3d.evaluation.backbone_batchnorm import (
    current_batch_without_updates, normalized_encoder_forward,
)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        self.eval_backbone_batchnorm_policy = "train_no_update"

    @normalized_encoder_forward
    def encode_batch(self, inputs):
        return self.backbone(inputs)


class BatchNormPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Adapter()
        self.inputs = torch.randn(32, 4) + 7

    def test_matches_existing_mask_policy_and_preserves_every_buffer(self):
        self.model.eval()
        before = copy.deepcopy(self.model.state_dict())
        reference = copy.deepcopy(self.model)
        reference.backbone[1].train()
        reference.backbone[1].momentum = 0
        expected = reference.encode_batch.__wrapped__(reference, self.inputs)
        actual = self.model.encode_batch(self.inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        self.assertTrue(all(not m.training for m in self.model.modules()))
        self.assertNotIn("forward", self.model.backbone[1].__dict__)

    def test_restores_forward_after_failure(self):
        self.model.eval()
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with current_batch_without_updates(self.model.backbone):
                self.model.backbone(self.inputs)
                raise RuntimeError("intentional")
        self.assertNotIn("forward", self.model.backbone[1].__dict__)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)

    def test_training_and_gradients_unchanged_after_evaluation(self):
        self.model.eval()
        self.model.encode_batch(self.inputs)
        self.model.train()
        reference = copy.deepcopy(self.model)
        torch.manual_seed(17)
        actual = self.model.encode_batch(self.inputs)
        torch.manual_seed(17)
        expected = reference.backbone(self.inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(self.model.backbone[0].weight.grad)
        self.assertEqual(self.model.backbone[1].num_batches_tracked.item(), 1)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)

    def test_frozen_training_keeps_saved_statistics_and_eval_keeps_modes(self):
        self.model.train()
        self.model.backbone.eval()
        expected = self.model.backbone(self.inputs)
        actual = self.model.encode_batch(self.inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.model.eval()
        self.model.encode_batch(self.inputs)
        self.assertTrue(all(not m.training for m in self.model.backbone.modules()))

    def test_refuses_accidentally_enabled_stochastic_layers(self):
        self.model.eval()
        self.model.backbone[2].train()
        with self.assertRaisesRegex(RuntimeError, "entirely eval-mode"):
            self.model.encode_batch(self.inputs)


if __name__ == "__main__":
    unittest.main()
