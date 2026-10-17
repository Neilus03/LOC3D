"""Native pretraining initialization keeps only exact encoder tensors."""
from types import SimpleNamespace

import pytest
import torch

from delimit3d.tasks.mask3d.initialization import load_litept_backbone_checkpoint


def payload():
    return {"model_state_dict": {
        "module.backbone.weight": torch.tensor([[1., 2.], [3., 4.]]),
        "module.backbone.bias": torch.tensor([5., 6.]),
        "module.projector.net.0.weight": torch.ones(4, 2),
        "module.criterion.log_temperature": torch.tensor(-2.3),
    }}


def load(tmp_path, value):
    path = tmp_path / "pretraining.pt"
    torch.save(value, path)
    model = SimpleNamespace(backbone=torch.nn.Linear(2, 2), decoder=torch.nn.Linear(2, 1))
    decoder_before = {key: tensor.clone() for key, tensor in model.decoder.state_dict().items()}
    report = load_litept_backbone_checkpoint(model, path)
    for key, tensor in decoder_before.items():
        assert torch.equal(tensor, model.decoder.state_dict()[key])
    return model, report


def test_native_temperature_is_excluded_and_encoder_is_exact(tmp_path):
    model, report = load(tmp_path, payload())
    assert torch.equal(model.backbone.weight, torch.tensor([[1., 2.], [3., 4.]]))
    assert torch.equal(model.backbone.bias, torch.tensor([5., 6.]))
    assert report["tensor_count"] == 2
    assert report["decoder_tensors_loaded"] == 0
    assert report["strict"] is True


@pytest.mark.parametrize("temperature", [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.ones(2)])
def test_invalid_temperature_is_rejected(tmp_path, temperature):
    value = payload()
    value["model_state_dict"]["module.criterion.log_temperature"] = temperature
    with pytest.raises(RuntimeError, match="one finite scalar"):
        load(tmp_path, value)


def test_task_decoder_state_is_never_silently_ignored(tmp_path):
    value = payload()
    value["model_state_dict"]["module.decoder.weight"] = torch.ones(1, 2)
    with pytest.raises(RuntimeError, match="refuses non-backbone state"):
        load(tmp_path, value)


def test_normalized_key_collision_is_rejected(tmp_path):
    value = payload()
    value["model_state_dict"]["backbone.bias"] = torch.zeros(2)
    with pytest.raises(RuntimeError, match="Duplicate normalized"):
        load(tmp_path, value)


def test_non_tensor_state_is_rejected(tmp_path):
    value = payload()
    value["model_state_dict"]["module.criterion.log_temperature"] = -2.3
    with pytest.raises(TypeError, match="Non-tensor checkpoint"):
        load(tmp_path, value)
