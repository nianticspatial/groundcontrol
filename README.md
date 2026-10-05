# Satellite-conditioned SEVA inference

Standalone inference code for the gated high-resolution satellite-conditioned SEVA checkpoint (`SevaSatRayAttnFusedQuerySeparateBlock`). It loads Metropolis and Argoverse, runs the same evaluation entrypoint as the training repo, and writes generated views plus metrics.

Weights and datasets stay outside this repository.

## Setup

```bash
mamba env create -f environment.yml
mamba activate sat-seva
pip install -e .
```

`environment.yml` installs PyTorch with CUDA 12.1. If that build does not match the machine, install PyTorch separately and then `pip install -r requirements.txt`.

## Weights

Set these paths in `src/osm_diffusion/configs/seva_training_raydir_sep_block_ca_metropolis_initcheck_hires_mlp_gated.yaml`, or keep the defaults if the files are already on disk:

| Config key | What it is |
| --- | --- |
| `paths.pretrained_weights_path` | Base SEVA `model.safetensors` directory |
| `paths.vae_ckpt_path` | Stable Diffusion 2.1 VAE directory |
| `paths.clip_hf_hub_path` | OpenCLIP cache (`ViT-H-14`, `laion2b_s32b_b79k`) |

The finetuned checkpoint is passed on the command line, not stored in the config. The Metropolis run below uses `ft_updated_svc_sat_ca_hires_gated_new_mlp/checkpoints/eval1.ckpt`.

Semantic metrics also expect Segment Anything weights at `/mnt/nas/shared/projects/sem_eval/sam_vit_h_4b8939.pth`.

## Data

Dataset roots are in:

- `src/osm_diffusion/configs/datasets/eval/metropolis.yaml`
- `src/osm_diffusion/configs/datasets/eval/argoverse.yaml`

Metropolis inference uses the precomputed DINOv3 satellite features in `dino_satellite_feats_dir`. Pass `--compute_dino_features` only if you want to encode satellite images on the fly; that path needs `DINOV3_REPO` and `DINOV3_WEIGHTS`.

## Metropolis inference

```bash
bash scripts/infer_metropolis.sh
```

Optional overrides: `CKPT`, `RES` (`W,H`), `OUTPUT_DIR`. Extra arguments are forwarded to `osm_diffusion.training.eval`.

Images are written under `$OUTPUT_DIR/metropolis_<resolution>_dual_cfg3_img2_baseline/metropolis/<sample_id>/`, and the metrics JSON is written in `$OUTPUT_DIR`.

Argoverse uses the same entrypoint with `--datasets argoverse`.
