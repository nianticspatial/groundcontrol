# osm_diffusion/training/eval.py
"""
Contains the core logic for model evaluation and a command-line interface
for evaluating a single checkpoint.

The `run_evaluation` function can be imported by other scripts.
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Tuple

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
from osm_diffusion.training.ema import EMAUnetAveraging


@dataclass
class EvaluationConfig:
    """A dataclass to hold all configuration for a single evaluation run."""

    checkpoint_path: str | None
    config_path: Path
    output_path: Path
    datasets: List["str"]  # name or path to dataset configs
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
    compute_dino_features: bool = False
    dual_cfg: bool = False
    sat_cfg_value: float = 2.0
    img_cfg_value: float = 2.0
    use_high_mem_gpu: bool = False
    # EMA configuration
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


def run_evaluation(cfg: EvaluationConfig, ds_id=None) -> List[Mapping[str, float]]:
    """
    Core, reusable function to evaluate a single checkpoint.

    Args:
        cfg: An EvaluationConfig object containing all necessary parameters.

    Returns:
        A list of dictionaries containing the validation metrics.
    """
    if cfg.checkpoint_path is None:
        print("No checkpoint path provided. Using the baseline SEVA model")
    print(f"--- [PID: {Path().owner()}] Running evaluation for: {cfg.checkpoint_path} ---")
    config = OmegaConf.load(cfg.config_path)

    print("Overriding dataset configurations...")
    config.data.datasets = cfg.datasets
    config.data.ds_config_path = cfg.datasets_config_path
    if config.get("val_data") is not None:
        config.val_data.datasets = cfg.datasets
    config.data.resolution = cfg.resolution
    config.model.img_h = cfg.resolution[1]
    config.model.img_w = cfg.resolution[0]
    if cfg.validation_debug_dir is not None:
        config.validation.save_validation_camera_specs = True

    # 1. Create validation dataloaders from the config
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
        conditioning_type = "cached features" if not cfg.compute_dino_features else "images"
        print(
            "Applying satellite perturbations "
            f"to {conditioning_type}: rotate={cfg.rotate_satellite}, "
            f"random_other={cfg.random_satellite}, flip={cfg.flip_satellite}, "
            f"zero={cfg.zero_satellite}"
        )
        val_datasets = [
            SatellitePerturbationDataset(
                dataset,
                satellite_perturbations,
                use_precomputed_features=not cfg.compute_dino_features,
            )
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
    # Prefetch factor for evaluation dataloaders
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

    # 2. Initialize the Lightning Module
    print("Initializing Lightning Module...")

    config.validation.denoising_steps = cfg.denoising_steps
    config.model.use_preprocessed_dino_features = not cfg.compute_dino_features
    config.model.use_dual_cfg = cfg.dual_cfg
    # config.model.sat_cfg_value = cfg.sat_cfg_value
    # config.model.img_cfg_value = cfg.img_cfg_value
    config.model.cfg_scale_image = cfg.img_cfg_value
    config.model.cfg_scale_sat = cfg.sat_cfg_value
    if cfg.reference_translation_noise_m is not None:
        config.model.reference_translation_noise_m = cfg.reference_translation_noise_m
    if cfg.reference_rotation_noise_deg is not None:
        config.model.reference_rotation_noise_deg = cfg.reference_rotation_noise_deg
    config.validation.semeval = True
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

    # 3. Setup and run the PyTorch Lightning Trainer
    print("Configuring PyTorch Lightning Trainer for validation...")

    # Setup EMA callback if enabled
    callbacks = []
    if cfg.use_ema:
        print(
            f"[Eval] Setting up EMA callback (device={cfg.ema_device}, decay={cfg.ema_decay}, trainable_only={cfg.ema_trainable_only})"
        )
        callbacks.append(
            EMAUnetAveraging(
                enabled=True,
                device=cfg.ema_device,
                every_n_steps=1,  # Not used during eval, only for training updates
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
        # "datasets_config": OmegaConf.to_container(datasets_config, resolve=True),
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


def get_evaluation_config():
    parser = argparse.ArgumentParser(description="Evaluate a SEVA model checkpoint.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=False,
        help="Names of the eval YAML configuration file.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        required=False,
        default=50,
        help="Number of denoising steps to use for evaluation.",
    )
    parser.add_argument(
        "--datasets-config-path",
        type=str,
        required=False,
        default="eval",
        help="Path to the folder containing the config file relative to configs/.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the model checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional: Path to save the results JSON file.",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for evaluation.")
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction of validation batches to evaluate.",
    )
    parser.add_argument(
        "--ds_mode",
        type=str,
        default="val",
        help="Dataset mode to use for evaluation (e.g., 'val', 'test').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--validation_debug_dir",
        type=str,
        default=None,
        help="Optional: Directory to save debug information during validation.",
    )
    parser.add_argument(
        "--pose_scale_factors",
        type=float,
        nargs="+",
        default=[2.0],
        help="Scale factors for the pose (e.g., '1.0 0.5 0.5').",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_total_frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use_precomputed_satellite_features",
        type=bool,
        default=None,
        help="Whether to use precomputed satellite features.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="576,576",
        help="Resolution to use for evaluation. Format: 'W,H'.",
    )
    parser.add_argument(
        "--compute_dino_features",
        action="store_true",
        help="Whether to compute DINO features on-the-fly during evaluation.",
    )
    parser.add_argument(
        "--use_dual_cfg",
        action="store_true",
        help="Whether to use dual configuration during evaluation.",
    )
    parser.add_argument(
        "--sat_cfg_value",
        type=float,
        default=2.0,
        help="Configuration value for the satellite model.",
    )
    parser.add_argument(
        "--img_cfg_value",
        type=float,
        default=2.0,
        help="Configuration value for the image model.",
    )
    parser.add_argument(
        "--use_high_mem_gpu",
        action="store_true",
        help="Whether to use a high memory GPU for evaluation.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA (Exponential Moving Average) weights from the checkpoint during evaluation.",
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default="cuda",
        help="Device to store EMA weights on ('cpu', 'cuda', or 'auto'). Default: 'cpu'",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate (not used during eval, but should match training config). Default: 0.999",
    )
    parser.add_argument(
        "--ema_trainable_only",
        action="store_true",
        help="Whether EMA was trained with trainable_only mode (memory efficient). Should match training config.",
    )
    parser.add_argument(
        "--reference_translation_noise_m",
        type=float,
        default=None,
        help="Optional override for reference-frame translation noise magnitude in meters.",
    )
    parser.add_argument(
        "--reference_rotation_noise_deg",
        type=float,
        default=None,
        help="Optional override for reference-frame yaw noise magnitude in degrees.",
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

    resolution = tuple(int(x) for x in args.resolution.split(","))
    assert len(resolution) <= 2, "Resolution must be in the format 'W,H'."
    if len(resolution) == 1:
        resolution = (resolution[0], resolution[0])

    ts = datetime.now().isoformat()
    ckpt_name = Path(args.checkpoint_path).stem if args.checkpoint_path else "seva"
    if args.output_path:
        output_path = args.output_path / f"{ckpt_name}_{ts}_eval_results.json"
    else:
        assert (
            args.checkpoint_path is not None
        ), "Checkpoint path must be provided when output path is not specified."
        output_path = Path(args.checkpoint_path).parent / f"{ckpt_name}_{ts}_eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create the configuration object for the core logic
    eval_config = EvaluationConfig(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        gpus=args.gpus,
        limit_val_batches=args.limit_val_batches,
        output_path=output_path,
        ds_mode=args.ds_mode,
        datasets=args.datasets,
        datasets_config_path=args.datasets_config_path,
        num_input_frames=args.num_input_frames,
        num_total_frames=args.num_total_frames,
        batch_size=args.batch_size,
        validation_debug_dir=args.validation_debug_dir,
        pose_scale_factors=args.pose_scale_factors,
        denoising_steps=args.denoising_steps,
        resolution=resolution,
        compute_dino_features=args.compute_dino_features,
        dual_cfg=args.use_dual_cfg,
        sat_cfg_value=args.sat_cfg_value,
        img_cfg_value=args.img_cfg_value,
        use_high_mem_gpu=args.use_high_mem_gpu,
        use_ema=args.use_ema,
        ema_device=args.ema_device,
        ema_decay=args.ema_decay,
        ema_trainable_only=args.ema_trainable_only,
        reference_translation_noise_m=args.reference_translation_noise_m,
        reference_rotation_noise_deg=args.reference_rotation_noise_deg,
        rotate_satellite=args.rotate_satellite,
        random_satellite=args.random_satellite,
        flip_satellite=args.flip_satellite,
        zero_satellite=args.zero_satellite,
    )
    return eval_config


def main():
    """Main entry point to run a single evaluation from the command line."""

    eval_config = get_evaluation_config()
    # 2. Call the evaluation function
    metrics = run_evaluation(eval_config)


if __name__ == "__main__":
    main()
