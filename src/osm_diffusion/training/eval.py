"""Run inference for one satellite-conditioned SEVA checkpoint and save images plus metrics.

Primary input is a lean inference YAML (see configs/infer_metropolis.yaml).
Optional CLI flags only override values already set in that config.
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from osm_diffusion.data import dataloader
from osm_diffusion.data.satellite_perturbations import (
    SatellitePerturbationConfig,
    SatellitePerturbationDataset,
)
from osm_diffusion.training import seva_lightning_trainer
from osm_diffusion.training.config_loader import load_inference_config
from osm_diffusion.training.ema import EMAUnetAveraging


@dataclass
class EvaluationConfig:
    """Resolved settings for a single evaluation run."""

    checkpoint_path: str | None
    config_path: Path
    output_path: Path
    datasets: List[str]
    num_input_frames: int = 1
    num_total_frames: int = 8
    gpus: int = 1
    pose_scale_factors: List[float] = field(default_factory=lambda: [2.0])
    validation_debug_dir: str | None = None
    limit_val_batches: float = 1.0
    ds_mode: str = "val"
    batch_size: int = 1
    datasets_config_path: str = "eval"
    denoising_steps: int = 50
    resolution: tuple[int, int] = (576, 576)
    dual_cfg: bool = False
    sat_cfg_value: float = 2.0
    img_cfg_value: float = 2.0
    use_high_mem_gpu: bool = False
    use_ema: bool = False
    ema_device: str = "cpu"
    ema_decay: float = 0.999
    ema_trainable_only: bool = False
    reference_translation_noise_m: float | None = None
    reference_rotation_noise_deg: float | None = None
    rotate_satellite: bool = False
    random_satellite: bool = False
    flip_satellite: bool = False
    zero_satellite: bool = False


def _as_resolution(value) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [int(x) for x in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [int(x) for x in value]
    else:
        parts = [int(value)]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) != 2:
        raise ValueError(f"Resolution must be int or (W,H), got {value!r}")
    return (parts[0], parts[1])


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def run_evaluation(cfg: EvaluationConfig, ds_id=None) -> List[Mapping[str, float]]:
    """Evaluate a single checkpoint using settings resolved from the inference config."""
    if cfg.checkpoint_path is None:
        print("No checkpoint path provided. Using the baseline SEVA model")
    print(f"--- [PID: {Path().owner()}] Running evaluation for: {cfg.checkpoint_path} ---")
    config = load_inference_config(cfg.config_path)

    print("Applying evaluation settings from config...")
    config.data.datasets = cfg.datasets
    config.data.ds_config_path = cfg.datasets_config_path
    if config.get("val_data") is not None:
        config.val_data.datasets = cfg.datasets
    config.data.resolution = list(cfg.resolution)
    config.model.img_h = cfg.resolution[1]
    config.model.img_w = cfg.resolution[0]
    if cfg.validation_debug_dir is not None:
        config.validation.save_validation_camera_specs = True

    print("Creating validation dataloaders...")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    val_datasets = dataloader.get_datasets(config.data, mode=cfg.ds_mode)
    satellite_perturbations = SatellitePerturbationConfig(
        rotate=cfg.rotate_satellite,
        random_other=cfg.random_satellite,
        flip=cfg.flip_satellite,
        zero=cfg.zero_satellite,
    )
    if satellite_perturbations.enabled:
        print(
            "Applying satellite perturbations to images: "
            f"rotate={cfg.rotate_satellite}, random_other={cfg.random_satellite}, "
            f"flip={cfg.flip_satellite}, zero={cfg.zero_satellite}"
        )
        val_datasets = [
            SatellitePerturbationDataset(dataset, satellite_perturbations)
            for dataset in val_datasets
        ]
    collator = dataloader.MultiViewCollator(
        max_input_views=cfg.num_input_frames,
        min_input_views=cfg.num_input_frames,
        max_views=cfg.num_total_frames,
        mode="val",
        shuffle_images=False,
        use_last_view_as_target_padding=False,
    )
    val_prefetch_factor = config.data.get("val_prefetch_factor", 2)
    val_dataloaders = [
        DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.data.val_num_workers,
            prefetch_factor=(
                val_prefetch_factor
                if config.data.val_num_workers and config.data.val_num_workers > 0
                else None
            ),
        )
        for ds in val_datasets
    ]

    print("Initializing Lightning Module...")

    config.validation.denoising_steps = cfg.denoising_steps
    config.model.use_dual_cfg = cfg.dual_cfg
    config.model.cfg_scale_image = cfg.img_cfg_value
    config.model.cfg_scale_sat = cfg.sat_cfg_value
    if cfg.reference_translation_noise_m is not None:
        config.model.reference_translation_noise_m = cfg.reference_translation_noise_m
    if cfg.reference_rotation_noise_deg is not None:
        config.model.reference_rotation_noise_deg = cfg.reference_rotation_noise_deg
    config.validation.semeval = True
    if config.get("lightning") is None:
        config.lightning = OmegaConf.create({"mixed_precision": "bf16-mixed"})
    lightning_model = seva_lightning_trainer.get_lightning_module(config)

    if cfg.validation_debug_dir is not None:
        lightning_model.validation_debug_dir = cfg.output_path.parent / cfg.validation_debug_dir
        lightning_model.save_validation_results = True
        lightning_model.save_validation_camera_specs = True
        print(f"Validation debug directory for images: {lightning_model.validation_debug_dir}")
    lightning_model.save_images_dir = cfg.output_path.parent / "eval_images"
    lightning_model.save_images_dir.mkdir(parents=True, exist_ok=True)
    lightning_model.save_validation_grid_images = False
    lightning_model.validation_scale_factors = cfg.pose_scale_factors
    lightning_model.garbage_collection_after_validation_step = True

    print("Configuring PyTorch Lightning Trainer for validation...")

    callbacks = []
    if cfg.use_ema:
        print(
            f"[Eval] Setting up EMA callback (device={cfg.ema_device}, decay={cfg.ema_decay}, "
            f"trainable_only={cfg.ema_trainable_only})"
        )
        callbacks.append(
            EMAUnetAveraging(
                enabled=True,
                device=cfg.ema_device,
                every_n_steps=1,
                start_after_steps=0,
                decay=cfg.ema_decay,
                trainable_only=cfg.ema_trainable_only,
            )
        )
    else:
        print("[Eval] EMA not enabled - using standard model weights")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        logger=False,
        limit_val_batches=cfg.limit_val_batches,
        precision=config.lightning.mixed_precision,
        callbacks=callbacks,
    )

    torch.cuda.empty_cache()
    trainer.strict_loading = False
    metrics = trainer.validate(
        model=lightning_model,
        dataloaders=val_dataloaders,
        ckpt_path=cfg.checkpoint_path,
        verbose=False,
    )

    print(f"--- [PID: {Path().owner()}] Finished evaluation for: {cfg.checkpoint_path} ---")

    results_data = {
        "checkpoint_path": cfg.checkpoint_path,
        "evaluation_timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "limit_val_batches": cfg.limit_val_batches,
        "batch_size": cfg.batch_size,
        "resolution": cfg.resolution,
        "satellite_perturbations": {
            "rotate": cfg.rotate_satellite,
            "random_other": cfg.random_satellite,
            "flip": cfg.flip_satellite,
            "zero": cfg.zero_satellite,
        },
    }

    with open(cfg.output_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"\nResults successfully saved to: {cfg.output_path}")

    print("\n--- Final Metrics ---")
    for metric_dict in metrics:
        for key, value in metric_dict.items():
            print(f"  {key}: {value:.4f}")
    print("---------------------\n")
    return metrics


def get_evaluation_config() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate a SEVA model from an inference YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the inference YAML (model, data, paths, dual-CFG, checkpoint).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional override for inference.checkpoint_path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional override for the results directory (inference.output_dir).",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Optional override for inference.limit_val_batches.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Optional override for inference.gpus.",
    )
    parser.add_argument(
        "--validation_debug_dir",
        type=str,
        default=None,
        help="Optional override for inference.validation_debug_dir.",
    )
    parser.add_argument(
        "--rotate_satellite",
        action="store_true",
        help="Randomly rotate each satellite input by 90 or 270 degrees.",
    )
    parser.add_argument(
        "--random_satellite",
        action="store_true",
        help="Use a satellite input from a different example in the same dataset.",
    )
    parser.add_argument(
        "--flip_satellite",
        action="store_true",
        help="Flip each satellite input horizontally.",
    )
    parser.add_argument(
        "--zero_satellite",
        action="store_true",
        help="Replace each satellite input with zeros.",
    )
    args = parser.parse_args()

    yaml_cfg = load_inference_config(args.config)
    infer = yaml_cfg.get("inference", OmegaConf.create({}))
    data = yaml_cfg.get("data", OmegaConf.create({}))
    model = yaml_cfg.get("model", OmegaConf.create({}))
    validation = yaml_cfg.get("validation", OmegaConf.create({}))

    checkpoint_path = _first_not_none(args.checkpoint_path, infer.get("checkpoint_path"))
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path must be set in paths.yaml or passed via --checkpoint-path."
        )

    output_dir = Path(
        _first_not_none(
            args.output_path,
            infer.get("output_dir"),
            Path(checkpoint_path).parent,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(checkpoint_path).stem
    ts = datetime.now().isoformat()
    output_path = output_dir / f"{ckpt_name}_{ts}_eval_results.json"

    resolution = _as_resolution(data.get("resolution", [576, 576]))
    datasets = list(data.get("datasets", ["metropolis"]))
    if not datasets:
        raise ValueError("data.datasets must list at least one dataset name.")

    pose_scale_factors = list(validation.get("pose_scale_factors", [2.0]))
    dual_cfg = bool(model.get("use_dual_cfg", False))
    sat_cfg_value = float(model.get("cfg_scale_sat", model.get("cfg_scale", 2.0)))
    img_cfg_value = float(model.get("cfg_scale_image", model.get("cfg_scale", 2.0)))

    return EvaluationConfig(
        checkpoint_path=str(checkpoint_path),
        config_path=args.config,
        output_path=output_path,
        datasets=datasets,
        datasets_config_path=str(data.get("ds_config_path", "eval")),
        num_input_frames=int(data.get("num_input_frames", data.get("max_input_views", 1))),
        num_total_frames=int(data.get("num_total_frames", data.get("max_views", 8))),
        batch_size=int(data.get("batch_size", 1)),
        resolution=resolution,
        gpus=int(_first_not_none(args.gpus, infer.get("gpus"), 1)),
        limit_val_batches=float(
            _first_not_none(args.limit_val_batches, infer.get("limit_val_batches"), 1.0)
        ),
        ds_mode=str(infer.get("ds_mode", "val")),
        validation_debug_dir=_first_not_none(
            args.validation_debug_dir, infer.get("validation_debug_dir")
        ),
        pose_scale_factors=pose_scale_factors,
        denoising_steps=int(validation.get("denoising_steps", 50)),
        dual_cfg=dual_cfg,
        sat_cfg_value=sat_cfg_value,
        img_cfg_value=img_cfg_value,
        use_ema=bool(infer.get("use_ema", False)),
        ema_device=str(infer.get("ema_device", "cpu")),
        ema_decay=float(infer.get("ema_decay", 0.999)),
        ema_trainable_only=bool(infer.get("ema_trainable_only", False)),
        reference_translation_noise_m=model.get("reference_translation_noise_m"),
        reference_rotation_noise_deg=model.get("reference_rotation_noise_deg"),
        rotate_satellite=bool(
            args.rotate_satellite or infer.get("rotate_satellite", False)
        ),
        random_satellite=bool(
            args.random_satellite or infer.get("random_satellite", False)
        ),
        flip_satellite=bool(args.flip_satellite or infer.get("flip_satellite", False)),
        zero_satellite=bool(args.zero_satellite or infer.get("zero_satellite", False)),
    )


def main():
    eval_config = get_evaluation_config()
    run_evaluation(eval_config)


if __name__ == "__main__":
    main()
