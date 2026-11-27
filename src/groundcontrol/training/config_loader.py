"""Load inference YAML and merge machine-local paths.yaml."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REQUIRED_PATH_KEYS = (
    "checkpoint_path",
    "pretrained_weights_path",
    "vae_ckpt_path",
    "clip_hf_hub_path",
    "data_dir",
    "satellite_dir",
    "dinov3_repo",
    "dinov3_weights",
)


def paths_yaml_for(config_path: Path) -> Path:
    return Path(config_path).resolve().parent / "paths.yaml"


def example_paths_yaml_for(config_path: Path) -> Path:
    return Path(config_path).resolve().parent / "paths.example.yaml"


def load_inference_config(config_path: Path | str) -> DictConfig:
    """Load an inference config and merge sibling ``paths.yaml``.

    ``paths.yaml`` should be created from ``paths.example.yaml`` and is gitignored.
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)

    paths_file = paths_yaml_for(config_path)
    example_file = example_paths_yaml_for(config_path)
    if not paths_file.exists():
        raise FileNotFoundError(
            f"Missing local paths file: {paths_file}\n"
            f"Copy the example and edit it:\n"
            f"  cp {example_file} {paths_file}"
        )

    local = OmegaConf.load(paths_file)
    missing = [key for key in REQUIRED_PATH_KEYS if local.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"paths.yaml is missing required keys: {', '.join(missing)}\n"
            f"See {example_file}"
        )

    overlay = OmegaConf.create(
        {
            "inference": {
                "checkpoint_path": local.checkpoint_path,
                "output_dir": local.get("output_dir", "outputs/metropolis"),
            },
            "paths": {
                "pretrained_weights_path": local.pretrained_weights_path,
                "vae_ckpt_path": local.vae_ckpt_path,
                "clip_hf_hub_path": local.clip_hf_hub_path,
                "dinov3_repo": local.dinov3_repo,
                "dinov3_weights": local.dinov3_weights,
                "dino_weights": local.get("dino_weights"),
                "sam_weights": local.get("sam_weights"),
            },
            "data": {
                "data_dir": local.data_dir,
                "satellite_dir": local.satellite_dir,
            },
        }
    )
    return OmegaConf.merge(cfg, overlay)
