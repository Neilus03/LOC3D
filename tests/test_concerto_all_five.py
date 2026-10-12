"""All-stage alignment, legacy parity, and real decoder gradient contracts."""
import pytest
import torch
from torch import nn

from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder
from delimit3d.models.concerto_wrapper import (
    ConcertoMaskAdapters, aggregate_encoder_features, agile_adapter, tensor_hash,
)
from delimit3d.models.litept_mask3d import LitePTMask3DDecoder, split_feature_pyramid
from delimit3d.models.litept_wrapper import LitePTFeaturePyramid, LitePTPyramidLevel
from delimit3d.training.concerto_contract import (
    ALL_FIVE_STAGE_RECIPE, THREE_STAGE_RECIPE, concerto_feature_recipe,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_fine_identity_and_ancestor_alignment_across_two_scenes():
    widths = (32, 64, 128, 256, 512)
    features = tuple(torch.arange(n).float()[:, None].expand(n, width) + stage * 100
                     for stage, (n, width) in enumerate(zip((8, 6, 4, 2, 2), widths)))
    maps = tuple(torch.tensor(values) for values in (
        [0, 1, 1, 2, 3, 3, 4, 5], [0, 0, 1, 2, 3, 3], [0, 0, 1, 1], [0, 1]))
    fused = aggregate_encoder_features(features, maps, recipe=ALL_FIVE_STAGE_RECIPE)
    assert tuple(x.shape[1] for x in fused) == (992, 960, 896, 768, 512)
    expected_ancestors = (
        list(range(8)), [0, 1, 1, 2, 3, 3, 4, 5],
        [0, 0, 0, 1, 2, 2, 3, 3], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
    for actual, source, ancestors in zip(fused[0].split(widths, dim=1), features, expected_ancestors):
        torch.testing.assert_close(actual, source[ancestors], rtol=0, atol=0)
    legacy = aggregate_encoder_features(features, maps)
    assert torch.equal(legacy[0][1], legacy[0][2])
    assert not torch.equal(fused[0][1], fused[0][2])
    # The deep descriptor is identical to the historical output; only the
    # formerly omitted 32+64 channels are added at the finest resolution.
    torch.testing.assert_close(fused[0][:, 96:], legacy[0], rtol=0, atol=0)
    torch.testing.assert_close(fused[1][:, 64:], legacy[1], rtol=0, atol=0)


def test_legacy_initial_weights_and_default_are_unchanged():
    assert concerto_feature_recipe({"concerto": {}}) == THREE_STAGE_RECIPE
    with pytest.raises(ValueError, match="Unknown Concerto feature recipe"):
        concerto_feature_recipe({"concerto": {"feature_recipe": "typo"}})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        historical_agile = nn.Linear(896, 72)
        torch.manual_seed(42)
        historical_mask = nn.Module()
        historical_mask.finest = nn.Linear(896, 72)
        historical_mask.memory = nn.ModuleList(nn.Linear(src, dst) for src, dst in
                                               zip((512, 768, 896, 896), (504, 252, 144, 72)))
    assert tensor_hash(agile_adapter()) == tensor_hash(historical_agile)
    assert tensor_hash(ConcertoMaskAdapters()) == tensor_hash(historical_mask)


@pytest.mark.parametrize("factory,count", [(agile_adapter, 71496), (ConcertoMaskAdapters, 722196)])
def test_all_five_pair_initialization_and_rng(factory, count):
    before = torch.get_rng_state().clone()
    first = factory(seed=42, feature_recipe=ALL_FIVE_STAGE_RECIPE)
    second = factory(seed=42, feature_recipe=ALL_FIVE_STAGE_RECIPE)
    assert torch.equal(torch.get_rng_state(), before)
    assert tensor_hash(first) == tensor_hash(second)
    assert sum(p.numel() for p in first.parameters()) == count


def synthetic_pyramid(trainable, feature_recipe=ALL_FIVE_STAGE_RECIPE):
    """Two scenes with overlapping coordinates and exact stride-2 parent maps."""
    torch.manual_seed(17)
    counts = (64, 32, 16, 8, 4)
    raw = tuple(torch.randn(2 * n, width, requires_grad=trainable)
                for n, width in zip(counts, (32, 64, 128, 256, 512)))
    maps = tuple(torch.arange(2 * n) // 2 for n in counts[:-1])
    fused = aggregate_encoder_features(raw, maps, recipe=feature_recipe)
    grids = tuple(torch.stack((torch.arange(n), torch.zeros(n, dtype=torch.long),
                               torch.zeros(n, dtype=torch.long)), dim=1).repeat(2, 1)
                  for n in counts)
    xyz = tuple(grid.float() * (.05 * 2**i) for i, grid in enumerate(grids))
    offsets = tuple(torch.tensor([n, 2 * n]) for n in counts)
    pyramid = LitePTFeaturePyramid(
        finest_features=fused[0], finest_grid=grids[0], finest_xyz=xyz[0], finest_offsets=offsets[0],
        memory_levels=tuple(LitePTPyramidLevel(
            name=f"concerto_enc{i}", stride=2**i, channels=fused[i].shape[1],
            features=fused[i], grid=grids[i], xyz=xyz[i], offsets=offsets[i]) for i in (4, 3, 2, 1)),
        point_xyz=xyz[0], point_offsets=offsets[0], original_to_finest=torch.arange(128),
        representative_indices=torch.arange(128), parent_maps=maps)
    return raw, pyramid


@pytest.mark.parametrize("task", ["agile3d", "mask3d"])
@pytest.mark.parametrize("trainable", [False, True])
def test_real_decoder_forward_backward_and_scene_boundaries(task, trainable):
    raw, pyramid = synthetic_pyramid(trainable)
    if task == "agile3d":
        adapters = agile_adapter(feature_recipe=ALL_FIVE_STAGE_RECIPE)
        decoder = Agile3DClickDecoder()
        adapted = adapters(pyramid.finest_features)
        losses = []
        for start, end in ((0, 64), (64, 128)):
            result = decoder(adapted[start:end], pyramid.finest_xyz[start:end],
                             clicks={0: [], 1: [0]}, click_times={0: [], 1: [0]})
            assert result["pred_masks"].shape == (64, 2)
            losses.append(result["pred_masks"].square().mean())
    else:
        adapters = ConcertoMaskAdapters(feature_recipe=ALL_FIVE_STAGE_RECIPE)
        decoder = LitePTMask3DDecoder(eval_queries=100)
        adapted = adapters(pyramid)
        assert adapted.memory_channels == (504, 252, 144, 72)
        assert adapted.memory_strides == (16, 8, 4, 2)
        losses = []
        for scene in split_feature_pyramid(adapted):
            assert torch.equal(scene.original_to_finest, torch.arange(64))
            result = decoder(scene, point_to_segment=torch.arange(64) // 2,
                             voxel_to_segment=torch.arange(64) // 2, num_segments=32)
            assert result["object_logits"].shape == (100, 2)
            assert result["mask_logits"].shape == (100, 64)
            losses.append(result["mask_logits"].square().mean() + result["object_logits"].square().mean())
    loss = sum(losses)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in adapters.parameters())
    if trainable:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in raw)
    else:
        assert all(p.grad is None for p in raw)
