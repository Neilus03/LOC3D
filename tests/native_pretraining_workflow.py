"""Opt-in two-GPU public pretraining and full checkpoint recovery check."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get("DELIMIT3D_NATIVE_CHECKS") != "1" or torch.cuda.device_count() != 2:
        raise RuntimeError("This check needs an isolated allocation with exactly two CUDA GPUs")
    from test_native_runtime import cloud
    from delimit3d.cli.prepare_sources import ProjectedProposalFrame, _array_record, _file_record, _write_raw_cache
    from delimit3d.data.cli import seal_main, fixed_eval_main, code_record_main
    from delimit3d.data.source_2d import SCHEMA
    from delimit3d.runtime.config import load_config
    from delimit3d.runtime.records import file_sha256, write_json
    from delimit3d.training.contracts import EXPERIMENT_ID, TWO_D_CELLS
    folder = Path(os.environ.get("DELIMIT3D_NATIVE_OUTPUT_DIR", ROOT / "runs" / "native_pretraining_check"))
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "sources"; source.mkdir()
    points, features = cloud()
    # Keep the synthetic mask planes separated through the normal scene
    # augmentation and voxelization; production survival gates remain active.
    points, features = (points * 4).numpy(), features.numpy()
    count = len(points); scenes = [f"scene_{index:05d}" for index in range(13)]
    for scene in scenes:
        destination = source / scene; destination.mkdir()
        arrays = {"points":points,"colors":np.uint8(features[:,:3]*255),"normals":features[:,3:],
                  "crop_indices":np.arange(count,dtype=np.int64)}
        for key,array in arrays.items():np.save(destination/f"{key}.npy",array)
        projected = {key:tuple(ProjectedProposalFrame(frame_id=f"0:{index}",split=split,
            eligible_global_indices=np.arange(count),proposal_global_indices=tuple(np.split(np.arange(count),16)),
            proposal_order_indices=np.arange(16),proposal_source_indices=np.arange(16),
            raw_stats={},used_relaxed_fallback=False) for index,split in enumerate(("train","heldout")))
            for key in ("g02","g05","g08")}
        _write_raw_cache(path=destination/"raw_projection_cache.npz",crop_indices=np.arange(count),
            projected=projected,num_points=count)
        artifacts={key:_array_record(destination/f"{key}.npy",relative_to=destination) for key in arrays}
        artifacts["raw_projection_cache"]=_file_record(destination/"raw_projection_cache.npz",relative_to=destination)
        write_json(destination/"source_manifest.json",{"schema_version":SCHEMA,"scene_id":scene,
            "dataset":"structured3d","experiment_id":EXPERIMENT_ID,"complete":True,"source_cells":list(TWO_D_CELLS),
            "labels":None,"crop":{"mode":"sphere_50k","num_points":count},"artifacts":artifacts,
            "frames":{"train":["0:0"],"heldout":["0:1"]},"counts":{name:{cell:16 for cell in TWO_D_CELLS}
            for name in ("trainable_proposals","heldout_trainable_proposals")}})
    train,val,audit,code,fixed=(folder/name for name in ("train.txt","val.txt","audit.json","code.json","fixed"))
    train.write_text("\n".join(scenes[:12])+"\n");val.write_text(scenes[12]+"\n")
    seal_main(["--source-root",str(source),"--train-split",str(train),"--holdout-split",str(val),"--output",str(audit)])
    fixed_eval_main(["--source-root",str(source),"--holdout-split",str(val),"--teacher","unsamv2",
        "--grid-size","0.05","--output",str(fixed)])
    code_record_main(["--output",str(code),"--repo-root",str(ROOT)])
    initial=folder/"initial.pt"
    subprocess.run([sys.executable,str(ROOT/"scripts/prepare_data.py"),"--action","backbone-init",
        "--litept-root",str(ROOT/".deps/LitePT"),"--litept-variant","litept_s_star",
        "--output",str(initial),"--seed","42","--grid-size","0.05","--input-features","rgbn6"],
        check=True,cwd=ROOT)
    recipe=load_config(ROOT/"configs/pretrain/base.yaml",expand_env=False)
    recipe.update(litept_root=str(ROOT/".deps/LitePT"),source_root=str(source),source_audit=str(audit),
        train_split=str(train),holdout_split=str(val),output_dir=str(folder/"continuous"),
        code_manifest=str(code),fixed_eval_pack=str(fixed),initial_checkpoint=str(initial),
        epochs=3,warmup_epochs=1,scenes_per_rank_per_update=6,grid_size=.05,required_crop_mode="sphere_50k",
        backbone_lr=.0001,grad_clip_norm=5,normalization_policy="sync_batchnorm",bn_recalibration_scenes=48,
        train_augmentation_profile="mask3d_scannet_rgbn6_sphere50k_v1",sampling_mode="coverage_multigranular_v2",
        hierarchy_supervision="query_mask",hierarchy_mask_loss_weight=.1,hierarchy_vicreg_loss_weight=0,multiscale_supervision=False,
        source_loading_mode="preload",evaluation_epochs="0,1,3",checkpoint_epochs="0,1,3")
    config=folder/"recipe.json";write_json(config,recipe)
    command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
             str(ROOT/"scripts/pretrain.py"),"--config",str(config)]
    subprocess.run(command,check=True,cwd=ROOT)
    continuous_summary=json.loads((folder/"continuous/run_summary.json").read_text())
    assert continuous_summary["mechanics_passed"] and all(continuous_summary["checks"].values())
    # Simulate an interrupted parent at epoch one. The continuation guard
    # requires its complete checkpoint history and rejects later files.
    parent=folder/"parent_at_epoch_0001"/"checkpoints"
    parent.mkdir(parents=True,exist_ok=False)
    for epoch in (0,1):
        name=f"epoch_{epoch:04d}.pt"
        shutil.copy2(folder/"continuous/checkpoints"/name,parent/name)
    checkpoint=parent/"epoch_0001.pt"
    subprocess.run(command+["--resume",str(checkpoint),"--resume-sha256",file_sha256(checkpoint),
        "--resume-into-fresh-attempt","--output-dir",str(folder/"recovered")],check=True,cwd=ROOT)
    recovered_summary=json.loads((folder/"recovered/run_summary.json").read_text())
    assert recovered_summary["mechanics_passed"] and all(recovered_summary["checks"].values())
    final=torch.load(folder/"continuous/checkpoints/epoch_0003.pt",map_location="cpu",weights_only=False)
    recovered=torch.load(folder/"recovered/checkpoints/epoch_0003.pt",map_location="cpu",weights_only=False)
    for key in ("model_state_dict","optimizer_state_dict"):
        def compare(a,b):
            if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:compare(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            else:assert a==b
        compare(final[key],recovered[key])
    assert final["update"]==recovered["update"]==3
    assert final["resumable"] and recovered["resumable"]
    assert len(recovered["rank_rng_states"])==len(recovered["rank_coverage_states"])==2
    write_json(folder/"COMPLETE.json",{"passed":True,"world_size":2,"global_batch":12,"updates":3,
        "normalization":"sync_batchnorm","checkpoint_recovery_verified":True,
        "model_and_optimizer_rtol":1e-5,"model_and_optimizer_atol":1e-6,
        "runtime_checks":len(recovered_summary["checks"]),
        "fixture_mask_loss_weight":.1,"auxiliary_gradient_budget_limit":.5})
    print(json.dumps({"passed":True,"public_pretraining_and_resume":True}),flush=True)


if __name__=="__main__":main()
