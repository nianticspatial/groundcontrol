import hashlib
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf

from groundcontrol.data.av2_dataset import ArgoverseDataset
from groundcontrol.data.metropolis_dataset import MetropolisDataset


def get_dataset_config(ds_configs_path: str, ds_name: str) -> DictConfig | ListConfig:
    config_path = (
        Path(__file__).parent.parent / "configs" / "datasets" / ds_configs_path / f"{ds_name}.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    return OmegaConf.load(config_path)


def get_datasets(datasets_config: DictConfig, mode: str):
    datasets = []
    ds_config_path = datasets_config.ds_config_path
    for ds_name_or_config in datasets_config.datasets:
        if isinstance(ds_name_or_config, str):
            ds_config = get_dataset_config(ds_config_path, ds_name_or_config)
            ds_name = ds_config.name
        else:
            ds_name = ds_name_or_config.name
            ds_config = ds_name_or_config
        ds_config.resolution = datasets_config.resolution
        if datasets_config.get("data_dir") is not None:
            ds_config.data_dir = datasets_config.data_dir
        if datasets_config.get("satellite_dir") is not None:
            ds_config.satellite_dir = datasets_config.satellite_dir
        if ds_name == "argoverse":
            ds = ArgoverseDataset(ds_config, mode=mode)
        elif ds_name == "metropolis":
            ds = MetropolisDataset(ds_config, mode=mode)
        else:
            raise ValueError(
                f"Unknown dataset name: {ds_name}. Supported datasets: metropolis, argoverse."
            )
        datasets.append(ds)
    return datasets


class MultiViewCollator:
    """
    Collates data from the dataset and prepares it for the model.

    This version pads or truncates every sample to a fixed `max_views` length,
    ensuring consistent tensor shapes across all batches.

    It performs three main tasks:
    1.  Ensures all samples have a fixed number of views (`max_views`).
    2.  Determines the roles of each view (input vs. target) randomly.
    3.  Assembles the batch with the required keys: `images`, `poses`, `intrinsics`,
        `input_mask`, and `target_mask`.

    Args:
        num_input_views (int): The number of views to be marked as input.
        max_views (int): The total number of views every sample will be padded/truncated to.
                            This will be the `N` dimension in the final batch tensors.
    """

    def __init__(
        self,
        max_input_views: int = 4,
        min_input_views: int = 2,
        max_views: int = 12,
        shuffle_images: bool = True,
        mode: str = "train",
        generator: torch.Generator = None,
        deterministic: bool = False,
        fixed_num_input_views: int | None = None,
        use_last_view_as_target_padding: bool = False,
        min_support_views: int = 0,
        max_support_views: int = 0,
    ):
        self.use_last_view_as_target_padding = use_last_view_as_target_padding
        if max_input_views >= max_views:
            raise ValueError(
                f"max_input_views ({max_input_views}) must be less than max_views ({max_views})."
            )
        self.max_input_views = max_input_views
        self.min_input_views = min_input_views
        self.max_views = max_views
        self.shuffle_images = shuffle_images
        self.deterministic = deterministic
        self.fixed_num_input_views = fixed_num_input_views
        if mode == "val" and generator is None and not deterministic:
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            random.seed(42)
        else:
            self.generator = generator
        self.mode = mode
        self.min_support_views = min_support_views
        self.max_support_views = max_support_views

    def __call__(self, batch_list: list[dict]) -> dict:
        """
        Processes a list of samples from the dataset to form a batch.
        """
        all_images, all_poses, all_intrinsics = [], [], []
        all_input_masks, all_target_masks = [], []
        all_gps_coords = []
        all_satellite_images = []
        all_reference_rotation = []
        all_reference_translation = []
        all_satellite_ppm = []
        all_support_masks = []

        all_sample_ids = []

        for sample in batch_list:
            # Build a per-sample deterministic generator in validation using sequence_key
            if self.deterministic:
                seq_key = sample.get("sample_id", None)
                # Mix base_seed, class tag and sequence_key for stability
                key_str = f"MultiViewCollator|{str(seq_key)}"
                digest = hashlib.sha256(key_str.encode("utf-8")).digest()
                sample_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
                generator = torch.Generator().manual_seed(sample_seed)
            else:
                generator = self.generator

            images, poses, intrinsics = (
                sample["images"],
                sample["poses"],
                sample["intrinsics"],
            )
            current_views = images.shape[0]
            if "gps_coords" in sample:
                gps_coords = sample["gps_coords"]
            else:
                gps_coords = None
            if "satellite_images" in sample:
                satellite_images = sample["satellite_images"]
                sat_images_are_per_view = len(satellite_images) == current_views
            else:
                satellite_images = None
                sat_images_are_per_view = False
            if "reference_rotation" in sample:
                reference_rotation = sample["reference_rotation"]
            else:
                reference_rotation = None
            if "reference_translation" in sample:
                reference_translation = sample["reference_translation"]
            else:
                reference_translation = None
            if "satellite_ppm" in sample:
                satellite_ppm = sample["satellite_ppm"]
            else:
                satellite_ppm = None

            # Shuffle the sequence of views
            # If more views than max_views, randomly select a subset
            if self.shuffle_images:
                selection_indices = torch.randperm(current_views, generator=self.generator)[
                    : self.max_views
                ]
            else:
                selection_indices = torch.arange(current_views).int()[: self.max_views]
            images = images[selection_indices]
            poses = poses[selection_indices]
            intrinsics = intrinsics[selection_indices]
            if satellite_images is not None and sat_images_are_per_view:
                satellite_images = satellite_images[selection_indices]
            current_views = min(current_views, self.max_views)
            if gps_coords is not None:
                gps_coords = gps_coords[selection_indices]

            if current_views < self.min_input_views:
                raise ValueError(
                    f"Sample {sample.get('scene_id', 'N/A')} has only {current_views} views after "
                    f"potential truncation, but `min_input_views` is set to {self.min_input_views}. "
                    f"Consider reducing `min_input_views` or increasing `max_views`."
                )

            if "sample_id" in sample:
                all_sample_ids.append(sample["sample_id"])
            else:
                all_sample_ids.append("")

            if "input_mask" in sample and "target_mask" in sample:
                input_mask = sample["input_mask"][selection_indices]
                target_mask = sample["target_mask"][selection_indices]
                support_mask = None
            else:
                num_input_views = random.randint(self.min_input_views, self.max_input_views)
                selection_indices = torch.randperm(current_views, generator=self.generator)

                input_indices = selection_indices[:num_input_views]
                target_indices = selection_indices[num_input_views:]

                input_mask = torch.zeros(current_views, dtype=torch.bool)
                input_mask[input_indices] = True

                target_mask = torch.zeros(current_views, dtype=torch.bool)
                target_mask[target_indices] = True

                if self.max_support_views > 0:
                    num_support_views = random.randint(
                        self.min_support_views, self.max_support_views
                    )
                    support_indices = target_indices[:num_support_views]
                    support_mask = torch.zeros(current_views, dtype=torch.bool)
                    support_mask[support_indices] = True
                else:
                    support_mask = None

            # --- Padding to max_views ---
            num_to_pad = self.max_views - current_views
            if num_to_pad > 0:
                # Pad data tensors
                if self.use_last_view_as_target_padding:
                    padding_image = images[-1:].repeat(num_to_pad, 1, 1, 1)
                    padding_pose = poses[-1:].repeat(num_to_pad, 1, 1)
                    padding_intrinsics = intrinsics[-1:].repeat(num_to_pad, 1, 1)
                    images = torch.cat([images, padding_image], dim=0)
                    poses = torch.cat([poses, padding_pose], dim=0)
                    intrinsics = torch.cat([intrinsics, padding_intrinsics], dim=0)

                    if gps_coords is not None:
                        padding_gps_coords = gps_coords[-1:].repeat(num_to_pad, 1, 1)
                        gps_coords = torch.cat([gps_coords, padding_gps_coords], dim=0)
                    if satellite_images is not None and sat_images_are_per_view:
                        padding_satellite_images = satellite_images[-1:].repeat(num_to_pad, 1, 1, 1)
                        satellite_images = torch.cat(
                            [satellite_images, padding_satellite_images], dim=0
                        )
                    input_mask = F.pad(input_mask, (0, num_to_pad), "constant", False)
                    target_mask = F.pad(target_mask, (0, num_to_pad), "constant", True)
                    if support_mask is not None:
                        support_mask = F.pad(support_mask, (0, num_to_pad), "constant", False)
                else:
                    padding_image = torch.zeros(num_to_pad, *images.shape[1:], dtype=images.dtype)

                    padding_pose = (
                        torch.eye(4, dtype=poses.dtype).unsqueeze(0).repeat(num_to_pad, 1, 1)
                    )

                    padding_intrinsics = (
                        torch.eye(3, dtype=intrinsics.dtype).unsqueeze(0).repeat(num_to_pad, 1, 1)
                    )
                    images = torch.cat([images, padding_image], dim=0)

                    poses = torch.cat([poses, padding_pose], dim=0)
                    intrinsics = torch.cat([intrinsics, padding_intrinsics], dim=0)

                    if gps_coords is not None:
                        padding_gps_coords = torch.zeros(
                            num_to_pad, *gps_coords.shape[1:], dtype=gps_coords.dtype
                        )
                        gps_coords = torch.cat([gps_coords, padding_gps_coords], dim=0)
                    if satellite_images is not None and sat_images_are_per_view:
                        padding_satellite_images = torch.zeros(
                            num_to_pad, *satellite_images.shape[1:], dtype=satellite_images.dtype
                        )
                        satellite_images = torch.cat(
                            [satellite_images, padding_satellite_images], dim=0
                        )

                    # Pad boolean masks with False
                    input_mask = F.pad(input_mask, (0, num_to_pad), "constant", False)
                    target_mask = F.pad(target_mask, (0, num_to_pad), "constant", False)
                    if support_mask is not None:
                        support_mask = F.pad(support_mask, (0, num_to_pad), "constant", False)

            # --- Append processed sample to lists ---
            all_images.append(images)
            all_poses.append(poses)
            all_intrinsics.append(intrinsics)
            all_input_masks.append(input_mask)
            all_target_masks.append(target_mask)
            if support_mask is not None:
                all_support_masks.append(support_mask)
            if gps_coords is not None:
                all_gps_coords.append(gps_coords)
            if satellite_images is not None:
                all_satellite_images.append(satellite_images)
            if reference_rotation is not None:
                all_reference_rotation.append(reference_rotation)
            if reference_translation is not None:
                all_reference_translation.append(reference_translation)
            if satellite_ppm is not None:
                all_satellite_ppm.append(satellite_ppm)

        # Stack all samples into final batch tensors
        batch = {
            "images": torch.stack(all_images),
            "camera_poses": torch.stack(all_poses),
            "intrinsics": torch.stack(all_intrinsics),
            "input_mask": torch.stack(all_input_masks),
            "target_mask": torch.stack(all_target_masks),
            "sample_id": all_sample_ids,
        }
        if all_gps_coords:
            batch["gps_coords"] = torch.stack(all_gps_coords)
        if all_satellite_images:
            batch["satellite_images"] = torch.stack(all_satellite_images)
        if all_reference_rotation:
            batch["reference_rotation"] = torch.stack(all_reference_rotation)
        if all_reference_translation:
            batch["reference_translation"] = torch.stack(all_reference_translation)
        if all_satellite_ppm:
            batch["satellite_ppm"] = torch.stack(all_satellite_ppm)
        if all_support_masks:
            batch["support_mask"] = torch.stack(all_support_masks)
        return batch

