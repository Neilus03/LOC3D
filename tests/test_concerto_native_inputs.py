"""Native-stage transfer, exact template reuse, and paired gradient boundaries."""
import copy

import pytest
import torch

from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder
from delimit3d.models.concerto_wrapper import configure_direct_decoder_inputs, tensor_hash
from delimit3d.models.litept_mask3d import LitePTMask3DDecoder, split_feature_pyramid
from delimit3d.training.concerto_contract import (
    DIRECT_INPUT_PROJECTION, LEGACY_INPUT_PROJECTION, NATIVE_STAGE_RECIPE,
    concerto_input_projection,
)
from test_concerto_all_five import synthetic_pyramid, bounded_cpu_threads


def test_native_memory_is_not_fused_and_finest_keeps_all_five_stages():
    raw, pyramid = synthetic_pyramid(True, NATIVE_STAGE_RECIPE)
    assert pyramid.memory_channels == (512, 256, 128, 64)
    for stage, level in zip((4, 3, 2, 1), pyramid.memory_levels):
        assert level.features is raw[stage]
    torch.testing.assert_close(pyramid.finest_features[:, :32], raw[0], rtol=0, atol=0)
    assert pyramid.finest_features.shape == (128, 992)


def test_native_mode_is_explicit_and_legacy_default_is_preserved():
    assert concerto_input_projection({"concerto": {}}) == LEGACY_INPUT_PROJECTION
    config = {"concerto": {"feature_recipe": NATIVE_STAGE_RECIPE}}
    with pytest.raises(ValueError, match="require direct"):
        concerto_input_projection(config)
    config["concerto"]["input_projection"] = DIRECT_INPUT_PROJECTION
    assert concerto_input_projection(config) == DIRECT_INPUT_PROJECTION


def test_input_initialization_does_not_reseed_cuda(monkeypatch):
    decoder = Agile3DClickDecoder()
    def unexpected_seed(*args, **kwargs):
        raise AssertionError("CPU projection initialization must not reseed CUDA")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_seed)
    configure_direct_decoder_inputs(decoder, task="agile3d", feature_recipe=NATIVE_STAGE_RECIPE)


@pytest.mark.parametrize("task,expected_count,projection_count", [
    ("agile3d", 1550336, 127104), ("mask3d", 1901698, 250496),
])
def test_paired_heads_preserve_every_non_input_tensor_and_rng(task, expected_count, projection_count):
    template = Agile3DClickDecoder() if task == "agile3d" else LitePTMask3DDecoder(eval_queries=100)
    original = copy.deepcopy(template.state_dict())
    pair = [copy.deepcopy(template), copy.deepcopy(template)]
    before = torch.get_rng_state().clone()
    reports = [configure_direct_decoder_inputs(m, task=task,
               feature_recipe=NATIVE_STAGE_RECIPE, seed=42) for m in pair]
    assert torch.equal(torch.get_rng_state(), before)
    assert reports[0] == reports[1]
    assert tensor_hash(pair[0]) == tensor_hash(pair[1])
    assert reports[0]["projection_parameters"] == projection_count
    assert sum(p.numel() for p in pair[0].parameters()) == expected_count
    for name, value in pair[0].state_dict().items():
        if name not in reports[0]["reinitialized_keys"]:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    restored = copy.deepcopy(pair[1])
    restored.load_state_dict(pair[0].state_dict(), strict=True)
    assert tensor_hash(restored) == tensor_hash(pair[0])


@pytest.mark.parametrize("task", ["agile3d", "mask3d"])
@pytest.mark.parametrize("trainable", [False, True])
def test_native_forward_backward_optimizer_and_evaluation(task, trainable):
    raw, pyramid = synthetic_pyramid(trainable, NATIVE_STAGE_RECIPE)
    decoder = Agile3DClickDecoder() if task == "agile3d" else LitePTMask3DDecoder(eval_queries=100)
    configure_direct_decoder_inputs(decoder, task=task, feature_recipe=NATIVE_STAGE_RECIPE)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    before = tensor_hash(decoder)

    def forward():
        results = []
        for scene in split_feature_pyramid(pyramid):
            if task == "agile3d":
                value = decoder(scene.finest_features, scene.finest_xyz,
                                clicks={0: [], 1: [0]}, click_times={0: [], 1: [0]})
                assert value["pred_masks"].shape == (64, 2)
                results.append(value["pred_masks"].square().mean())
            else:
                value = decoder(scene, point_to_segment=torch.arange(64) // 2,
                                voxel_to_segment=torch.arange(64) // 2, num_segments=32)
                assert value["object_logits"].shape == (100, 2)
                assert value["mask_logits"].shape == (100, 64)
                results.append(value["mask_logits"].square().mean() + value["object_logits"].square().mean())
        return sum(results)

    loss = forward()
    assert torch.isfinite(loss)
    loss.backward()
    for value in raw:
        if trainable:
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
        else:
            assert value.grad is None
    inputs = ([decoder.input_adapter] if task == "agile3d" else
              [decoder.mask_features_head, *decoder.memory_projections])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for module in inputs for p in module.parameters())
    optimizer.step()
    assert tensor_hash(decoder) != before
    decoder.eval()
    with torch.no_grad():
        assert torch.isfinite(forward())
