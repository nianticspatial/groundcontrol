import hashlib
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.data
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from osm_diffusion.data.av2_dataset import ArgoverseDataset
from osm_diffusion.data.metropolis_dataset import MetropolisDataset


def get_dataset_config(ds_configs_path: str, ds_name: str) -> DictConfig | ListConfig:
    config_path = (
        Path(__file__).parent.parent / "configs" / "datasets" / ds_configs_path / f"{ds_name}.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    ds_config = OmegaConf.load(config_path)
    return ds_config


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
        if ds_name == "argoverse":
            ds = ArgoverseDataset(ds_config, mode=mode)
        elif ds_name == "metropolis":
            ds = MetropolisDataset(ds_config, mode=mode)
        else:
            raise ValueError(
                f"Unknown dataset name: {ds_name}. This release includes metropolis and argoverse."
            )
        datasets.append(ds)
    return datasets


class MultiDatasetSampler(torch.utils.data.Sampler):
    """
    A custom sampler that draws from multiple datasets using shuffled queues,
    with support for a torch.Generator for reproducible seeding.
    """

    def __init__(self, datasets: list, probabilities: list, generator=None):
        if len(datasets) != len(probabilities):
            raise ValueError("Length of datasets and probabilities must match.")

        self.datasets = datasets
        self.probabilities = torch.tensor(list(probabilities))
        self.probabilities = self.probabilities / self.probabilities.sum()
        # Expected maximum number of samples per epoch across all datasets
        self.total_samples = int(max([len(d) / p for p, d in zip(probabilities, datasets)]))
        self.generator = generator
        print(f"MultiDatasetSampler will draw {self.total_samples} samples per epoch.")

        self.shuffled_indices_iters = []

    def __iter__(self):
        # This method is called once at the beginning of each epoch.

        # 1. Create freshly shuffled queues using the provided generator.
        self.shuffled_indices_iters = [
            iter(torch.randperm(len(d), generator=self.generator).tolist()) for d in self.datasets
        ]

        # 2. The main sampling loop
        for _ in range(self.total_samples):
            # Choose a dataset using the generator.
            dataset_idx = torch.multinomial(self.probabilities, 1, generator=self.generator).item()

            # 3. Draw the next sample index from the shuffled queue.
            try:
                sample_idx = next(self.shuffled_indices_iters[dataset_idx])
            except StopIteration:
                # If a queue is exhausted, reshuffle it using the generator.
                self.shuffled_indices_iters[dataset_idx] = iter(
                    torch.randperm(
                        len(self.datasets[dataset_idx]), generator=self.generator
                    ).tolist()
                )
                sample_idx = next(self.shuffled_indices_iters[dataset_idx])

            yield (dataset_idx, sample_idx)

    def __len__(self):
        # Using this to define an epoch is not ideal, as datasets may have different lengths.
        return self.total_samples


class CombinedDataset(torch.utils.data.Dataset):
    """A dataset that wraps multiple datasets and accesses them via (dataset_idx, sample_idx)."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, idx_tuple):
        dataset_idx, sample_idx = idx_tuple
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class SceneInterleavingSampler(torch.utils.data.Sampler):
    """
    Deterministic sampler that interleaves samples across scenes/logs in a round-robin fashion.

    Assumes the dataset exposes `samples` where each entry's first element is a scene/log id
    (e.g., (log_id, timestamp)). Falls back gracefully if this structure is absent.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.index_sequence = self._build_index_sequence()

    def _build_index_sequence(self):
        # If dataset has no attribute `samples`, return sequential indices deterministically
        if not hasattr(self.dataset, "samples"):
            return list(range(len(self.dataset)))

        samples = getattr(self.dataset, "samples")
        # Group indices by scene/log id using the first element when possible
        scene_to_indices = {}
        for idx, sample in enumerate(samples):
            try:
                scene_id = sample[0]
            except Exception:
                scene_id = "__single_group__"
            scene_to_indices.setdefault(scene_id, []).append(idx)

        # Sort scene ids for determinism (convert tuples to repr for consistent ordering)
        def _scene_sort_key(s):
            try:
                return (0, str(s))
            except Exception:
                return (1, repr(s))

        ordered_scenes = sorted(scene_to_indices.keys(), key=_scene_sort_key)

        # Within each scene, keep original order (assumed timestamp-sorted in dataset)
        per_scene_iters = [iter(scene_to_indices[s]) for s in ordered_scenes]

        # Round-robin across scenes
        result = []
        remaining = True
        while remaining:
            remaining = False
            for it in per_scene_iters:
                try:
                    result.append(next(it))
                    remaining = True
                except StopIteration:
                    continue
        return result

    def __iter__(self):
        # Deterministic fixed order
        return iter(self.index_sequence)

    def __len__(self):
        return len(self.index_sequence)


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
        all_satellite_features = []
        all_osm_images = []
        all_height_maps = []
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
            if "satellite_features" in sample:
                satellite_features = sample["satellite_features"]
                sat_features_are_per_view = len(satellite_features) == current_views
            else:
                satellite_features = None
                sat_features_are_per_view = False
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
            if "osm_images" in sample:
                osm_images = sample["osm_images"]
                osm_images_are_per_view = len(osm_images) == current_views
            else:
                osm_images = None
                osm_images_are_per_view = False
            if "height_maps" in sample:
                height_maps = sample["height_maps"]
                height_maps_are_per_view = len(height_maps) == current_views
            else:
                height_maps = None
                height_maps_are_per_view = False

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
            if satellite_features is not None and sat_features_are_per_view:
                satellite_features = satellite_features[selection_indices]
            if osm_images is not None and osm_images_are_per_view:
                osm_images = osm_images[selection_indices]
            current_views = min(current_views, self.max_views)
            if gps_coords is not None:
                gps_coords = gps_coords[selection_indices]

            if current_views < self.min_input_views:
                raise ValueError(
                    f"Sample {sample.get('scene_id', 'N/A')} has only {current_views} views after "
                    f"potential truncation, but `min_input_views` is set to {self.min_input_views}. "
                    f"Consider reducing `min_input_views` or increasing `max_views`."
                )

            # --- Create Input/Target Masks for the available views ---
            # view_indices = list(range(current_views))
            # random.shuffle(view_indices)

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
                # TODO: this shuffling is not needed, we can use range(current_views)[:num_input_views],
                # but this means that all input views are consecutive, which may not be good for training.
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
                    if satellite_features is not None and sat_features_are_per_view:
                        padding_satellite_features = satellite_features[-1:].repeat(
                            num_to_pad, 1, 1, 1
                        )
                        satellite_features = torch.cat(
                            [satellite_features, padding_satellite_features], dim=0
                        )
                    if osm_images is not None and osm_images_are_per_view:
                        padding_osm_images = osm_images[-1:].repeat(num_to_pad, 1, 1, 1)
                        osm_images = torch.cat([osm_images, padding_osm_images], dim=0)
                    if height_maps is not None and height_maps_are_per_view:
                        padding_height_maps = height_maps[-1:].repeat(num_to_pad, 1, 1, 1)
                        height_maps = torch.cat([height_maps, padding_height_maps], dim=0)
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
                    if satellite_features is not None and sat_features_are_per_view:
                        padding_satellite_features = torch.zeros(
                            num_to_pad,
                            *satellite_features.shape[1:],
                            dtype=satellite_features.dtype,
                        )
                        satellite_features = torch.cat(
                            [satellite_features, padding_satellite_features], dim=0
                        )
                    if osm_images is not None and osm_images_are_per_view:
                        padding_osm_images = torch.zeros(
                            num_to_pad, *osm_images.shape[1:], dtype=osm_images.dtype
                        )
                        osm_images = torch.cat([osm_images, padding_osm_images], dim=0)
                    if height_maps is not None and height_maps_are_per_view:
                        padding_height_maps = torch.zeros(
                            num_to_pad, *height_maps.shape[1:], dtype=height_maps.dtype
                        )
                        height_maps = torch.cat([height_maps, padding_height_maps], dim=0)

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
            if satellite_features is not None:
                all_satellite_features.append(satellite_features)
            if reference_rotation is not None:
                all_reference_rotation.append(reference_rotation)
            if osm_images is not None:
                all_osm_images.append(osm_images)
            if reference_translation is not None:
                all_reference_translation.append(reference_translation)
            if satellite_ppm is not None:
                all_satellite_ppm.append(satellite_ppm)
            if height_maps is not None:
                all_height_maps.append(height_maps)

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
        if all_satellite_features:
            batch["satellite_features"] = torch.stack(all_satellite_features)
        if all_reference_rotation:
            batch["reference_rotation"] = torch.stack(all_reference_rotation)
        if all_reference_translation:
            batch["reference_translation"] = torch.stack(all_reference_translation)
        if all_satellite_ppm:
            batch["satellite_ppm"] = torch.stack(all_satellite_ppm)
        if all_osm_images:
            batch["osm_images"] = torch.stack(all_osm_images)
        if all_height_maps:
            batch["height_maps"] = torch.stack(all_height_maps)
        if all_support_masks:
            batch["support_mask"] = torch.stack(all_support_masks)
        return batch


def create_dataloader(
    dataset_config,
    mode,
    max_input_views,
    min_input_views,
    num_views_in_batch,
    batch_size,
    shuffle=False,
    num_workers=4,
    min_support_views=0,
    max_support_views=0,
):
    datasets = get_datasets(dataset_config, mode)
    print(f"Found {len(datasets)} datasets")

    collator = MultiViewCollator(
        max_input_views=max_input_views,
        min_input_views=min_input_views,
        max_views=num_views_in_batch,
        shuffle_images=(mode == "train"),
        mode=mode,
        min_support_views=min_support_views,
        max_support_views=max_support_views,
    )
    combined_dataset = ConcatDataset(datasets)

    # Prefetch factor controls how many batches each worker prefetches ahead
    # Higher values = more memory usage but better GPU utilization
    prefetch_factor = (
        dataset_config.get("prefetch_factor", 4) if hasattr(dataset_config, "get") else 4
    )

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Shuffle is True for ConcatDataset, not needed for custom sampler
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def create_train_concat_dataloader(dataset_config, shuffle: bool = True):
    return create_dataloader(
        dataset_config,
        mode="train",
        max_input_views=dataset_config.max_input_views,
        min_input_views=dataset_config.min_input_views,
        num_views_in_batch=dataset_config.max_views,
        batch_size=dataset_config.batch_size,
        shuffle=shuffle,
        num_workers=dataset_config.num_workers,
        min_support_views=dataset_config.get("min_support_views", 0),
        max_support_views=dataset_config.get("max_support_views", 0),
    )


def create_train_prob_sampler_dataloader(
    dataset_config,
    use_ray=False,
    base_seed=42,
):
    # When using a distributed strategy, this function runs on each worker.
    # We can get the worker's global rank to create a unique seed.
    if use_ray and torch.distributed.is_initialized():
        worker_rank = torch.distributed.get_rank()
        worker_seed = base_seed + worker_rank
        generator = torch.Generator().manual_seed(worker_seed)
        print(f"INFO: Worker {worker_rank} initialized generator with seed {worker_seed}")
    else:  # Fallback for local or single-process training
        generator = torch.Generator().manual_seed(base_seed)

    # 1. --- Setup Data ---
    print("Setting up dataset and dataloader...")
    datasets = get_datasets(dataset_config, mode="train")

    collator = MultiViewCollator(
        max_input_views=dataset_config.max_input_views,
        min_input_views=dataset_config.min_input_views,
        max_views=dataset_config.max_views,
        shuffle_images=True,
        mode="train",
        min_support_views=dataset_config.get("min_support_views", 0),
        max_support_views=dataset_config.get("max_support_views", 0),
        # generator=generator,
    )

    sampler = MultiDatasetSampler(
        datasets=datasets,
        probabilities=dataset_config.train_dataset_probabilities,
        # generator=generator,
    )

    combined_train_dataset = CombinedDataset(datasets)

    # Prefetch factor controls how many batches each worker prefetches ahead
    # Higher values = more memory usage but better GPU utilization
    # Default is 2, but 4-8 can significantly speed up I/O-bound data loading
    prefetch_factor = dataset_config.get("prefetch_factor", 4)

    return DataLoader(
        combined_train_dataset,
        batch_size=dataset_config.batch_size,
        # shuffle=True,
        sampler=sampler,
        collate_fn=collator,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor if dataset_config.num_workers > 0 else None,
    )
