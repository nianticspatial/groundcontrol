# GroundControl

[GroundControl](https://nianticspatial.github.io/groundcontrol/) is a satellite-condtioned multi-view diffusion model. When generating novel views, it uses layout context from the satellite image, when the target viewpoint has low coverage in the input viewpoints. 

## Setup

```bash
mamba env create -f environment.yml
mamba activate groundcontrol
pip install -e .
```

If the CUDA build in `environment.yml` does not match your machine, install PyTorch separately, then `pip install -r requirements.txt`.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GC_CHECKPOINT` | yes | Fine-tuned checkpoint (`.ckpt`) |
| `GC_PRETRAINED_WEIGHTS` | yes | Base SEVA / SVC weights directory |
| `GC_VAE_PATH` | yes | SD 2.1 VAE directory |
| `GC_CLIP_HF_HUB` | yes | Local Hugging Face hub cache for CLIP |
| `METROPOLIS_DATA_DIR` | yes | Metropolis dataset root |
| `METROPOLIS_SATELLITE_DIR` | yes | Preprocessed satellite images |
| `DINOV3_REPO` | yes | Local DINOv3 torch.hub checkout |
| `DINOV3_WEIGHTS` | yes | DINOv3 ViT-L/16 weights file |
| `GC_OUTPUT_DIR` | no | Results directory (default `outputs/metropolis`) |
| `GC_DINO_WEIGHTS` | no | DINOv1 weights for semantic metrics |
| `GC_SAM_WEIGHTS` | no | Segment Anything weights for semantic metrics |

## Run

All run settings are in `src/osm_diffusion/configs/infer_metropolis.yaml`
(model `satellite_ray_guided_ca`, dual CFG sat=3 / img=2).

```bash
export GC_CHECKPOINT=...
export GC_PRETRAINED_WEIGHTS=...
export GC_VAE_PATH=...
export GC_CLIP_HF_HUB=...
export METROPOLIS_DATA_DIR=...
export METROPOLIS_SATELLITE_DIR=...
export DINOV3_REPO=...
export DINOV3_WEIGHTS=...

python -m osm_diffusion.training.eval \
  --config src/osm_diffusion/configs/infer_metropolis.yaml
```

Optional overrides: `--checkpoint-path`, `--output-path`, `--limit-val-batches`.

```bash
bash scripts/infer_metropolis.sh
bash scripts/infer_metropolis.sh --limit-val-batches 1
```

`CKPT` and `OUTPUT_DIR` override checkpoint and output directory for the shell wrapper.
