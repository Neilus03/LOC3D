"""Opt-in CUDA checks using pinned sources and checkpoints, never research runs."""
import gc
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DELIMIT3D_NATIVE_CHECKS") != "1" or not torch.cuda.is_available(),
                               reason="Opt-in native CUDA verification")


def cloud():
    axes = torch.arange(16, dtype=torch.float32)
    points = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1).reshape(-1, 3) * .2
    generator = torch.Generator().manual_seed(42)
    features = torch.rand((len(points), 6), generator=generator)
    features[:, 3:] = torch.nn.functional.normalize(features[:, 3:], dim=1)
    return points, features


@pytest.fixture(autouse=True)
def cleanup():
    torch.set_num_threads(2)
    yield
    gc.collect()
    torch.cuda.empty_cache()


def task_config(task, name, tmp_path, frozen):
    from delimit3d.runtime.config import load_config
    from delimit3d.data.cli import decoder_main
    from delimit3d.runtime.records import config_sha256
    config = load_config(ROOT / f"configs/tasks/{task}_{name}.yaml", expand_env=False)
    template = tmp_path / f"{task}.pt"
    decoder_main(["--task", task, "--output", str(template)])
    source = ROOT / ".deps"
    config["model"]["litept_root"] = str(source / "LitePT")
    if name != "litept":
        config[name]["source_root"] = str(source / ("Concerto" if name == "concerto" else "Sonata"))
        config[name]["checkpoint"] = str(source / "weights" / (name + "_small.pth"))
    if task == "mask3d":
        config["initialization"].update(backbone_source="scratch", backbone_checkpoint=None,
            freeze_backbone=frozen, decoder_template=str(template))
    else:
        if name == "litept":
            config["encoder_initialization"].update(kind="scratch", checkpoint=None)
        config["model"]["freeze_encoder"] = frozen
    config["_config_sha256"] = config_sha256(config)
    return config, {"litept_root": source / "LitePT", "decoder_initialization": template}


@pytest.mark.parametrize("name", ["litept", "concerto", "sonata"])
@pytest.mark.parametrize("task", ["mask3d", "agile3d"])
@pytest.mark.parametrize("frozen", [False, True])
def test_task_native_forward_loss_optimizer_and_freezing(name, task, frozen, tmp_path):
    from delimit3d.tasks.agile3d.engine import state_sha256
    config, paths = task_config(task, name, tmp_path, frozen)
    points, features = cloud()
    count = len(points)
    points = points.repeat(2, 1).cuda(); features = features.repeat(2, 1).cuda()
    offsets = torch.tensor([count, count * 2], device="cuda")
    if task == "mask3d":
        from delimit3d.tasks.mask3d.factory import build_model
        from delimit3d.tasks.mask3d.losses.mask3d_segment_loss import Mask3DSegmentSetCriterion
        from delimit3d.tasks.mask3d.data.segment_targets import build_segment_instance_targets
        model, _ = build_model(config, torch.device("cuda"))
        groups = model.parameter_groups(backbone_lr=1e-5, decoder_lr=1e-4)
        optimizer = torch.optim.AdamW(groups)
        model.train()
        before = state_sha256(model.backbone.state_dict())
        segments = (torch.arange(count, device="cuda") // 8).repeat(2)
        predictions = model(points, features, segments, scene_offsets=offsets)
        truth = torch.where(torch.arange(count, device="cuda") < count // 2, 1, 2)
        targets = [build_segment_instance_targets(truth, segments[:count])] * 2
        losses = Mask3DSegmentSetCriterion()(predictions, targets)
        loss = losses["loss_total"]
    else:
        from delimit3d.tasks.agile3d.model import Agile3DModel
        from delimit3d.tasks.agile3d.trainer import optimizer_for
        from delimit3d.evaluation.agile3d_decoder import compute_agile3d_losses
        model = Agile3DModel(config, paths).cuda().train()
        optimizer = optimizer_for(model, config)
        before = state_sha256(model.backbone.state_dict())
        grid = torch.floor(points / .05).int()
        tokens, xyz = model.encode_batch(grid, points, features, offsets)
        losses = []
        for feature, coord in zip(tokens, xyz):
            clicks = {0: [], 1: [0]}
            result = model.decode(feature, coord, clicks, {0: [], 1: [0]})
            truth = torch.zeros(len(feature), dtype=torch.long, device="cuda")
            truth[:len(feature) // 2] = 1
            losses.append(compute_agile3d_losses(result, truth, scene_xyz=coord, clicks=clicks)[0])
        loss = sum(losses)
    assert torch.isfinite(loss)
    if name == "litept":
        rope = [m for m in model.backbone.modules() if m.__class__.__name__ == "PointROPE"]
        assert rope and all(m.__class__.__module__.endswith("pointrope_cuda") for m in rope)
    loss.backward()
    decoder_gradients = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert decoder_gradients and all(torch.isfinite(gradient).all() for gradient in decoder_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in decoder_gradients) > 0
    backbone_gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert bool(backbone_gradients) != frozen
    if backbone_gradients:
        assert all(torch.isfinite(gradient).all() for gradient in backbone_gradients)
    optimizer.step()
    assert (state_sha256(model.backbone.state_dict()) == before) == frozen
    del model, optimizer, loss


@pytest.mark.parametrize("recipe", ["sam2", "multiscale", "hierarchy"])
def test_native_pretraining_objectives(recipe, tmp_path):
    from delimit3d.pretraining.model import LitePTContrastiveModel
    from delimit3d.pretraining.sources import SceneSource, RawFrame
    from delimit3d.pretraining.sampling import training_batch
    from delimit3d.pretraining.settings import sampling_config_for_profile
    points, features = cloud()
    count = len(points)
    masks = (np.arange(count // 2), np.arange(count // 2, count))
    keys = ("sam2",) if recipe == "sam2" else ("g02", "g05", "g08")
    frames = tuple(RawFrame(granularity=key, physical_frame_id="0:0", split="train",
        visible_indices=np.arange(count), proposal_offsets=np.array([0,count//2,count]),
        proposal_point_indices=np.concatenate(masks)) for key in keys)
    scene = SceneSource(scene_id="scene_00000", manifest_path=tmp_path / "source.json", manifest_sha256="a"*64,
        points=points.numpy(), features=features.numpy(), labels_by_key={}, raw_frames=frames,
        colors=np.uint8(features[:,:3].numpy()*255), normals=features[:,3:].numpy(), crop_mode="full_scene")
    cells = tuple("2d/" + key for key in keys)
    grid_size = .02 if recipe != "hierarchy" else .05
    groups, batch, cell = None, None, None
    if recipe == "multiscale":
        cell, batch = training_batch(scene=scene, epoch=21, points=points.cuda(), device=torch.device("cuda"),
            sampling=sampling_config_for_profile("pf_hnm_512_p16_v1"), cells=cells)
    else:
        from delimit3d.data.contrastive_sampler_v2 import FrameProposalCatalog, sample_multigranular_frame_group_plan
        catalogs = {"2d/" + f.granularity: (FrameProposalCatalog("2d/" + f.granularity,
            f.physical_frame_id, torch.as_tensor(f.visible_indices, device="cuda"), torch.as_tensor(f.proposal_offsets, device="cuda"),
            torch.as_tensor(f.proposal_point_indices, device="cuda")),) for f in frames}
        plan = sample_multigranular_frame_group_plan(scene_id=scene.scene_id, epoch=0, seed=42,
            points=points.cuda(), frames_by_cell=catalogs, cells=cells, total_proposal_quota=2*len(cells),
            grid_size=grid_size, require_hierarchy_routes=recipe=="hierarchy",
            require_negative_proposal_membership=True)
        groups = plan.groups
    model = LitePTContrastiveModel(litept_root=ROOT / ".deps/LitePT", grid_size=grid_size,
        token_pure_dec0=recipe != "multiscale", token_pure_cells=cells,
        multiscale_supervision=recipe == "multiscale", hierarchy_supervision=recipe == "hierarchy").cuda().train()
    result = model(points.cuda(), features.cuda(), batch, source_cell=cell, frame_groups=groups,
        supervision_step=21, supervision_total_steps=100, hierarchy_ramp_fraction=1.)
    loss = result["loss_total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.backbone.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def task_fixture(task, root):
    from plyfile import PlyData, PlyElement
    from delimit3d.runtime.records import file_sha256, write_json
    points, features = cloud()
    points, features = points.numpy(), features.numpy()
    count = len(points)
    labels = np.where(np.arange(count) < count//2, 1, 2).astype(np.int64)
    scans, normals, targets = root/"scans", root/"normals", root/"targets"
    scans.mkdir(); normals.mkdir(); targets.mkdir()
    rows = {}
    scenes = [f"scene000{i}_00" for i in range(3)]
    for scene in scenes:
        if task == "mask3d":
            folder = scans/scene; folder.mkdir()
            arrays = {"points":points, "colors":np.uint8(features[:,:3]*255),
                      "normals":features[:,3:]}
            for key,array in arrays.items(): np.save(folder/f"{key}.npy",array)
            write_json(folder/"scene_meta.json", {"schema":"delimit3d_scannet_geometry/v1",
                "scene_id":scene,"num_points":count,"granularities":[.5],"label_files":{},
                "artifacts_sha256":{key:file_sha256(folder/f"{key}.npy") for key in arrays}})
            np.savez(targets/f"{scene}.npz",instance_ids=labels, segment_ids=np.arange(count)//8,
                instance_class_pairs=np.array([[1,4],[2,5]],dtype=np.int64))
            rows[scene] = {"cache_file":f"{scene}.npz","num_points":count,
                           "sha256":file_sha256(targets/f"{scene}.npz")}
        else:
            vertex = np.zeros(count,dtype=[("x","f4"),("y","f4"),("z","f4"),
                ("R","u1"),("G","u1"),("B","u1"),("label","i4")])
            for index,key in enumerate(("x","y","z")):vertex[key]=points[:,index]
            for index,key in enumerate(("R","G","B")):vertex[key]=np.uint8(features[:,index]*255)
            vertex["label"]=labels
            PlyData([PlyElement.describe(vertex,"vertex")]).write(scans/f"{scene}.ply")
            folder=normals/scene; folder.mkdir()
            np.save(folder/"normal.npy",features[:,3:]); np.save(folder/"coord.npy",points)
    train, val = root/"train.txt", root/"val.txt"
    if task == "mask3d":
        train.write_text("\n".join(scenes[:2])+"\n");val.write_text(scenes[2]+"\n")
        write_json(targets/"manifest.json",{"cache_version":1,"eval_benchmark":"scannet20","scenes":rows})
    else:
        train=root/"train.json";val=root/"val.json"
        write_json(train,{scene+"_obj_2":None for scene in scenes[:2]})
        write_json(val,{scenes[2]+"_obj_2":None})
    return scans,normals,targets,train,val


@pytest.mark.parametrize("task",["mask3d","agile3d"])
def test_public_task_train_evaluate_and_resume(task,tmp_path):
    from delimit3d.tasks.cli import train_main,evaluate_main
    from delimit3d.runtime.records import write_json
    config,paths=task_config(task,"litept",tmp_path,True)
    scans,normals,targets,train,val=task_fixture(task,tmp_path)
    if task == "mask3d":
        config["data"].update(scans_root=str(scans),train_split=str(train),val_split=str(val),
            scannet_gt_cache_root=str(targets),augmentations=False,preload=False,preload_workers=0)
        config["train"].update(max_epochs=2,batch_scenes_per_step=2,num_workers=0,eval_every_epochs=1,save_every_epochs=1)
        config["initialization"]["decoder_template"]=str(paths["decoder_initialization"])
        checkpoint_name,first_name="latest.pt","epoch_0001.pt"
    else:
        config["paths"].update(litept_root=str(paths["litept_root"]),scan_folder=str(scans),normal_root=str(normals),
            train_list=str(train),val_list=str(val),decoder_initialization=str(paths["decoder_initialization"]),
            single_object_root=None)
        config["protocol"].update(epochs=2,batch_size=2,val_epochs=1,lr_drop=[1],max_num_clicks=20)
        config["runtime"]["num_workers"]=0
        checkpoint_name,first_name="checkpoint.pth","checkpoint0001.pth"
    recipe=tmp_path/"recipe.json";write_json(recipe,config)
    continuous,recovered,evaluated=tmp_path/"continuous",tmp_path/"recovered",tmp_path/"evaluation"
    train_main(["--config",str(recipe),"--output",str(continuous)])
    final=torch.load(continuous/"checkpoints"/checkpoint_name,map_location="cpu",weights_only=False)
    first=continuous/"checkpoints"/first_name
    assert final["validation_completed"]
    with pytest.raises(FileExistsError):
        train_main(["--config",str(recipe),"--output",str(continuous)])
    evaluate_main(["--config",str(recipe),"--checkpoint",str(continuous/"checkpoints"/checkpoint_name),
        "--output",str(evaluated),"--protocol","mo"])
    result=json.loads(next(evaluated.glob("evaluation_epoch_*.json")).read_text())
    assert result["epoch"]==2
    train_main(["--config",str(recipe),"--resume",str(first),"--output",str(recovered)])
    resumed=torch.load(recovered/"checkpoints"/checkpoint_name,map_location="cpu",weights_only=False)
    key="model_state_dict" if task=="mask3d" else "model"
    for name,value in final[key].items():
        torch.testing.assert_close(resumed[key][name],value,rtol=1e-5,atol=1e-6)
    optimizer_key="optimizer_state_dict" if task=="mask3d" else "optimizer"
    for index,state in final[optimizer_key]["state"].items():
        for name,value in state.items():
            torch.testing.assert_close(resumed[optimizer_key]["state"][index][name],value,rtol=1e-5,atol=1e-6)
    assert json.loads((recovered/"status.json").read_text())["complete"]
