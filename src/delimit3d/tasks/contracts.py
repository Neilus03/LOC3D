"""The model interface must stay fixed when evaluating a new dataset."""
from delimit3d.models.backbones import backbone_name, ptv3_recipe, ptv3_direct_inputs


def model_contract(config):
    name = backbone_name(config)
    model = {key: value for key, value in config["model"].items()
             if key not in {"litept_root", "freeze_encoder", "backbone_name", "litept_variant"}}
    # FPN selection changes the readout even when tensor widths agree.
    result = {"task": config["task"], "backbone": name, "model": model}
    if config["task"] == "agile3d":
        result.update(voxel_size_m=config["protocol"]["voxel_size_m"],
                      input_features=config["protocol"]["input_features"])
    if name != "litept_s_star":
        result.update(feature_recipe=ptv3_recipe(config), direct_inputs=ptv3_direct_inputs(config))
    return result
