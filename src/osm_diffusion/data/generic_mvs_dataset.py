import abc
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)


class GenericMVSDataset(Dataset, abc.ABC):
    """
    An abstract base class for multi-view, multi-sensor autonomous driving datasets.
    It provides a generic structure for fetching samples, supporting various
    sampling and image processing strategies.
    """

    def __init__(self, config: Dict, mode: str = "train"):
        super().__init__()
        self.mode = mode
        if self.mode == "val":
            self.mode = config.get("val_mode", "val")
        self.config = config
        resolution = config.get("resolution", 576)
        # Support resolution as int (square) or tuple/list/ListConfig (width, height) for non-square
        if isinstance(resolution, (int, float)):
            self.resolution = (int(resolution), int(resolution))
        elif isinstance(resolution, (tuple, list, ListConfig)):
            # Convert to list first to handle ListConfig, then to tuple
            resolution_list = list(resolution) if not isinstance(resolution, tuple) else resolution
            if len(resolution_list) == 2:
                self.resolution = (int(resolution_list[0]), int(resolution_list[1]))
            else:
                raise ValueError(
                    f"resolution must be int or tuple/list of (width, height), got {resolution} with length {len(resolution_list)}"
                )
        else:
            raise ValueError(
                f"resolution must be int or tuple/list/ListConfig of (width, height), got {resolution} of type {type(resolution)}"
            )
        self.resolution_width, self.resolution_height = self.resolution
        self.min_views = config.get("min_views", 1)
        self.use_gps = config.get("use_gps", False)
        self.single_sample_overfit = config.get("single_sample_overfit", False)

        self.cropping_method = config.get("cropping_method", "center")
        if self.cropping_method not in ["center", "dual"]:
            raise ValueError(f"Invalid cropping_method: {self.cropping_method}")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.use_temporal_subsampling = config.get("use_temporal_subsampling", False)
        self.num_subsample_cameras = config.get("num_subsample_cameras", 3)
        self.min_subsample_cameras = config.get("min_subsample_cameras", self.num_subsample_cameras)
        self.max_subsample_cameras = config.get("max_subsample_cameras", self.num_subsample_cameras)
        self.temporal_window_size = config.get("temporal_window_size", 5)
        self.temporal_window_input_mask = config.get("temporal_window_input_mask", None)
        self.num_max_images = config.get("num_max_images", 15)
        self.satellite_features_only_mode = config.get("satellite_features_only_mode", False)
        if self.use_temporal_subsampling and self.temporal_window_size % 2 == 0:
            raise ValueError("temporal_window_size must be an odd number.")
        if self.temporal_window_input_mask is not None:
            if len(self.temporal_window_input_mask) != self.temporal_window_size:
                raise ValueError(
                    "temporal_window_input_mask length must match temporal_window_size. "
                    f"Got mask length={len(self.temporal_window_input_mask)} and "
                    f"temporal_window_size={self.temporal_window_size}."
                )
            if not all(v in (0, 1, False, True) for v in self.temporal_window_input_mask):
                raise ValueError("temporal_window_input_mask values must be binary (0/1 or bool).")
            self.temporal_window_input_mask = [bool(v) for v in self.temporal_window_input_mask]
            if not any(self.temporal_window_input_mask):
                raise ValueError("temporal_window_input_mask must mark at least one input timestamp.")

        # --- Satellite Map Configurations ---
        self.use_satellite = config.get("use_satellite", False)
        self.sat_features_pca_basis = None
        self.sat_features_pca_mean = None
        self.sat_features_pca_dim = None
        if self.use_satellite:
            self.satellite_dir = Path(config.get("satellite_dir"))
            self.satellite_configs = config.get("satellite_configs", [])
            if not self.satellite_dir or not self.satellite_dir.exists():
                raise FileNotFoundError(f"Satellite directory not found at: {self.satellite_dir}")
            if not self.satellite_configs:
                raise ValueError(
                    "'satellite_configs' must be provided when 'use_satellite' is True."
                )
            self.dino_satellite_feats_dir = config.get("dino_satellite_feats_dir", None)
            if (
                self.dino_satellite_feats_dir is not None
                and Path(self.dino_satellite_feats_dir).exists()
            ):
                print(
                    f"DINO satellite features directory found at: {self.dino_satellite_feats_dir}"
                )
            elif self.dino_satellite_feats_dir is not None:
                raise FileNotFoundError(
                    f"DINO satellite features directory not found at: {self.dino_satellite_feats_dir}"
                )

            sat_features_pca_basis_path = config.get("satellite_features_pca_basis_path", None)
            sat_features_pca_num_components = config.get(
                "satellite_features_pca_num_components", None
            )
            if sat_features_pca_basis_path is not None:
                basis_path = Path(sat_features_pca_basis_path)
                if not basis_path.exists():
                    raise FileNotFoundError(
                        f"Satellite features PCA basis file not found at: {basis_path}"
                    )

                with np.load(basis_path, allow_pickle=True) as pca_data:
                    if "basis" not in pca_data or "mean" not in pca_data:
                        raise ValueError(
                            f"PCA basis file must contain 'basis' and 'mean' keys: {basis_path}"
                        )
                    basis = np.asarray(pca_data["basis"], dtype=np.float32)
                    mean = np.asarray(pca_data["mean"], dtype=np.float32)

                if basis.ndim != 2:
                    raise ValueError(
                        f"Expected PCA basis with shape [feature_dim, num_bases], got {basis.shape}"
                    )
                if mean.ndim != 1:
                    raise ValueError(
                        f"Expected PCA mean with shape [feature_dim], got {mean.shape}"
                    )
                if basis.shape[0] != mean.shape[0]:
                    raise ValueError(
                        f"PCA basis/mean dimension mismatch: basis={basis.shape}, mean={mean.shape}"
                    )

                max_available_bases = basis.shape[1]
                if sat_features_pca_num_components is None:
                    proj_dim = max_available_bases
                else:
                    proj_dim = int(sat_features_pca_num_components)
                    if proj_dim < 1:
                        raise ValueError(
                            f"satellite_features_pca_num_components must be >= 1, got {proj_dim}"
                        )
                    if proj_dim > max_available_bases:
                        raise ValueError(
                            f"Requested {proj_dim} PCA components, but basis has only {max_available_bases}"
                        )

                self.sat_features_pca_basis = torch.from_numpy(basis[:, :proj_dim])
                self.sat_features_pca_mean = torch.from_numpy(mean)
                self.sat_features_pca_dim = proj_dim
                print(
                    "Satellite feature PCA projection enabled: "
                    f"path={basis_path}, input_dim={basis.shape[0]}, output_dim={proj_dim}"
                )
            if self.satellite_features_only_mode and self.dino_satellite_feats_dir is None:
                raise ValueError(
                    "satellite_features_only_mode requires dino_satellite_feats_dir to be set."
                )

        self.sample_cameras_per_item = config.get("sample_cameras_per_item", 0)

        self.use_osm = config.get("use_osm", False)
        if self.use_osm:
            self.osm_dir = Path(config.osm_dir)
            self.osm_configs = config.osm_configs
        else:
            self.osm_dir = None
            self.osm_configs = []

        self.use_height_maps = config.get("use_height_maps", False)
        if self.use_height_maps:
            self.height_maps_dir = Path(config.height_maps_dir)
            self.height_map_configs = config.height_map_configs
        else:
            self.height_maps_dir = None
            self.height_map_configs = []

        # --- Input/Target Camera Configuration ---
        self.input_camera_names = config.get("input_camera_names", None)
        self.target_camera_names = config.get("target_camera_names", None)

        self.samples = []
        self._build_samples()
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.samples)} samples.")

    @abc.abstractmethod
    def _build_samples(self):
        """(Abstract) Implemented by child class to build the list of sample keys."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_camera_names(self, log_id) -> List[Any]:
        """(Abstract) Returns the list of all camera identifiers for the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_timestamps_for_log(self, log_id: str) -> List[int]:
        """(Abstract) Returns a list of all timestamps for a given log/scene ID."""
        raise NotImplementedError

    @abc.abstractmethod
    def _process_view(self, view_identifier: Dict) -> Optional[Dict[str, List]]:
        """(Abstract) Processes a single view from its identifier dict."""
        raise NotImplementedError

    @abc.abstractmethod
    def _poses_to_gps(self, poses: torch.Tensor, log_id: str) -> torch.Tensor:
        """(Abstract) Converts a batch of camera poses to GPS coordinates."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_satellite_views(
        self,
        log_id: str,
        timestamp: Any,
        reference_rotation: Optional[torch.Tensor],
        views_w_T_cam: Optional[List[torch.Tensor]],
        intrinsics: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """(Abstract) Loads all configured satellite views for a given sample."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_satellite_view_features(
        self,
        log_id: str,
        timestamp: Any,
    ) -> Optional[List[torch.Tensor]]:
        """(Abstract) Loads all preprocessed satellite view features for a given sample."""
        raise NotImplementedError

    def _get_osm_views(
        self,
        log_id: str,
        timestamp: Any,
        reference_rotation: Optional[torch.Tensor],
        views_w_T_cam: Optional[List[torch.Tensor]],
        intrinsics: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """(Abstract) Loads all configured OSM views for a given sample."""
        if self.use_osm:
            raise NotImplementedError
        else:
            return None

    def _get_height_maps(
        self,
        log_id: str,
        timestamp: Any,
        reference_rotation: Optional[torch.Tensor],
        views_w_T_cam: Optional[List[torch.Tensor]],
        intrinsics: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """(Abstract) Loads all configured OSM views for a given sample."""
        if self.use_height_maps:
            raise NotImplementedError
        else:
            return None

    @abc.abstractmethod
    def _get_reference_rotation(
        self, log_id: str, timestamp: Any, poses: list[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """(Abstract) Returns the 3x3 reference rotation matrix for a given sample."""
        raise NotImplementedError

    def _get_reference_translation(
        self, log_id: str, timestamp: Any, poses: list[torch.Tensor]
    ) -> torch.Tensor:
        """(Abstract) Returns the 3D reference translation vector for a given sample."""
        t = torch.zeros(3)
        return t

    @staticmethod
    def resize_and_crop(
        image: Image.Image, intrinsics: np.ndarray, resolution: Union[int, Tuple[int, int]]
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Resize and crop image to target resolution.

        Uses max scale ratio to ensure both dimensions are at least as large as target
        after scaling, then center-crops to exact target size. This is suitable for
        downscaling scenarios where the source image is larger than the target.

        Args:
            image: Input PIL Image
            intrinsics: Camera intrinsics matrix (3x3 numpy array)
            resolution: Target resolution as int (square) or tuple (width, height)

        Returns:
            Cropped image and updated intrinsics
        """
        # Normalize resolution to (width, height)
        if isinstance(resolution, (int, float)):
            target_width = target_height = int(resolution)
        elif isinstance(resolution, (tuple, list, ListConfig)):
            resolution_list = list(resolution) if not isinstance(resolution, tuple) else resolution
            if len(resolution_list) == 2:
                target_width, target_height = int(resolution_list[0]), int(resolution_list[1])
            else:
                raise ValueError(
                    f"resolution must be int or tuple/list of (width, height), got {resolution} with length {len(resolution_list)}"
                )
        else:
            raise ValueError(
                f"resolution must be int or tuple/list/ListConfig of (width, height), got {resolution} of type {type(resolution)}"
            )

        original_width, original_height = image.size
        # Use max scale ratio to ensure both dimensions are >= target after scaling
        # This ensures we can always crop to the target size
        scale_ratio = max(target_width / original_width, target_height / original_height)
        scaled_width = int(round(original_width * scale_ratio))
        scaled_height = int(round(original_height * scale_ratio))
        resized_image = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)

        # Center crop to target resolution
        crop_x_start = (scaled_width - target_width) // 2
        crop_y_start = (scaled_height - target_height) // 2
        crop_box = (
            crop_x_start,
            crop_y_start,
            crop_x_start + target_width,
            crop_y_start + target_height,
        )
        cropped_image = resized_image.crop(crop_box)
        updated_intrinsics = intrinsics.copy()
        updated_intrinsics[0, :] *= scale_ratio
        updated_intrinsics[1, :] *= scale_ratio
        updated_intrinsics[0, 2] -= crop_x_start
        updated_intrinsics[1, 2] -= crop_y_start
        return cropped_image, updated_intrinsics

    @staticmethod
    def create_two_square_images(
        image: Image.Image, intrinsics: np.ndarray, resolution: Union[int, Tuple[int, int]]
    ) -> Tuple[Image.Image, np.ndarray, Image.Image, np.ndarray]:
        """
        Create two cropped images from a single image (dual cropping method).
        For non-square resolutions, creates two images of the target resolution.

        Uses max scale ratio to ensure both dimensions are at least as large as target
        after scaling, then crops from both ends (left/right or top/bottom depending
        on aspect ratio). This is suitable for downscaling scenarios.

        Args:
            image: Input PIL Image
            intrinsics: Camera intrinsics matrix (3x3 numpy array)
            resolution: Target resolution as int (square) or tuple (width, height)

        Returns:
            Two cropped images and their corresponding intrinsics
        """
        # Normalize resolution to (width, height)
        if isinstance(resolution, (int, float)):
            target_width = target_height = int(resolution)
        elif isinstance(resolution, (tuple, list, ListConfig)):
            resolution_list = list(resolution) if not isinstance(resolution, tuple) else resolution
            if len(resolution_list) == 2:
                target_width, target_height = int(resolution_list[0]), int(resolution_list[1])
            else:
                raise ValueError(
                    f"resolution must be int or tuple/list of (width, height), got {resolution} with length {len(resolution_list)}"
                )
        else:
            raise ValueError(
                f"resolution must be int or tuple/list/ListConfig of (width, height), got {resolution} of type {type(resolution)}"
            )

        original_width, original_height = image.size
        # Use max scale ratio to ensure both dimensions are >= target after scaling
        # This ensures we can always crop to the target size
        scale_ratio = max(target_width / original_width, target_height / original_height)
        scaled_width = int(round(original_width * scale_ratio))
        scaled_height = int(round(original_height * scale_ratio))
        resized_image = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[0, :] *= scale_ratio
        scaled_intrinsics[1, :] *= scale_ratio

        if scaled_width > scaled_height:
            # Crop from left and right
            image1 = resized_image.crop((0, 0, target_width, target_height))
            intrinsics1 = scaled_intrinsics.copy()
            crop2_x_start = scaled_width - target_width
            image2 = resized_image.crop((crop2_x_start, 0, scaled_width, target_height))
            intrinsics2 = scaled_intrinsics.copy()
            intrinsics2[0, 2] -= crop2_x_start
        else:
            # Crop from top and bottom
            image1 = resized_image.crop((0, 0, target_width, target_height))
            intrinsics1 = scaled_intrinsics.copy()
            crop2_y_start = scaled_height - target_height
            image2 = resized_image.crop((0, crop2_y_start, target_width, scaled_height))
            intrinsics2 = scaled_intrinsics.copy()
            intrinsics2[1, 2] -= crop2_y_start
        return image1, intrinsics1, image2, intrinsics2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Tuple]]:
        if self.single_sample_overfit:
            index = 0
        if self.satellite_features_only_mode:
            return self._get_satellite_features_only_item(index)
        if self.use_temporal_subsampling:
            return self._get_temporal_item(index)
        else:
            return self._get_single_timestamp_item(index)

    def _get_satellite_features_only_item(
        self, index: int
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        sample_entry = self.samples[index]
        if len(sample_entry) < 2:
            raise ValueError(
                f"Expected sample tuple with at least (log_id, timestamp), got: {sample_entry}"
            )
        log_id = sample_entry[0]
        center_timestamp = sample_entry[1]

        satellite_features = self._get_satellite_view_features(log_id, center_timestamp)
        satellite_features = self._project_satellite_features(satellite_features)
        if satellite_features is None or len(satellite_features) == 0:
            raise ValueError(
                f"No satellite features returned for sample {(log_id, center_timestamp)}."
            )

        return {
            "satellite_features": torch.stack(satellite_features),
            "sample_id": (log_id, center_timestamp, 0),
        }

    def _get_input_target_mask(self, num_images) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Creates input and target masks based on input/target camera configuration.
        Returns None if input/target cameras are not configured or if overridden by child class.

        Note: For datasets with input/target cameras, the mask is typically calculated
        directly in _get_single_timestamp_item() and passed to _package_data().
        This method is kept for backward compatibility with child classes that may override it.
        """
        return None

    def _package_data(
        self,
        data_lists: Dict,
        sample_id_tuple: Tuple,
        reference_rotation: Optional[torch.Tensor],
        reference_translation: Optional[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        """Final data packaging, converting lists of tensors to batched tensors."""
        # Build masks from is_input flags if available
        input_target_mask = None
        num_images = len(data_lists.get("images", []))
        if "is_input" in data_lists and data_lists["is_input"]:
            is_input_list = data_lists["is_input"]
            # Remove is_input from data_lists so it doesn't get stacked
            del data_lists["is_input"]
            if any(is_input_list) and self.mode in ["val", "test"]:
                input_mask = torch.tensor(is_input_list, dtype=torch.bool)
                input_target_mask = (input_mask, ~input_mask)

        packaged = {key: torch.stack(val) for key, val in data_lists.items() if val}
        packaged["sample_id"] = sample_id_tuple

        aux_data = self._get_auxiliary_data(sample_id_tuple)
        if aux_data:
            packaged.update(aux_data)

        if self.use_gps:
            log_id = sample_id_tuple[0]
            packaged["gps_coords"] = self._poses_to_gps(packaged["poses"], log_id)
        if reference_rotation is not None:
            packaged["reference_rotation"] = reference_rotation
        if reference_translation is not None:
            packaged["reference_translation"] = reference_translation
        if self.use_satellite and self.satellite_configs:
            # Expose satellite scale so downstream modules can convert metric offsets to pixels.
            packaged["satellite_ppm"] = torch.tensor(
                float(self.satellite_configs[0].get("ppm", 4.0)), dtype=torch.float32
            )
        if self.mode == "val" or self.mode == "test":
            # Use mask from is_input flags if available, otherwise fall back to _get_input_target_mask
            if input_target_mask is None:
                input_target_mask = self._get_input_target_mask(num_images)
            if input_target_mask is not None:
                packaged["input_mask"] = input_target_mask[0]
                packaged["target_mask"] = input_target_mask[1]
        return packaged

    def _get_auxiliary_data(self, sample_id_tuple: Tuple) -> Optional[Dict[str, Any]]:
        """(Abstract) Returns a dictionary of auxiliary, non-tensor data for a sample."""
        return {}

    def _project_satellite_features(
        self, satellite_features: Optional[List[torch.Tensor]]
    ) -> Optional[List[torch.Tensor]]:
        if satellite_features is None:
            return None
        if self.sat_features_pca_basis is None:
            return satellite_features

        projected_features = []
        in_dim = int(self.sat_features_pca_mean.shape[0])
        out_dim = int(self.sat_features_pca_basis.shape[1])
        mean = self.sat_features_pca_mean
        basis = self.sat_features_pca_basis

        for features in satellite_features:
            if features.shape[-1] != in_dim:
                raise ValueError(
                    f"Satellite feature dim mismatch. Expected {in_dim}, got {features.shape[-1]}"
                )
            reshaped = features.to(torch.float32).reshape(-1, in_dim)
            projected = (reshaped - mean) @ basis
            projected_features.append(projected.reshape(*features.shape[:-1], out_dim))
        return projected_features

    def _get_single_timestamp_item(self, index: int) -> Dict[str, Union[torch.Tensor, Tuple]]:
        log_id, center_timestamp = self.samples[index]
        data_lists = {"images": [], "poses": [], "intrinsics": [], "timestamps": []}

        # Track is_input flags if input/target cameras are configured
        use_input_target = (
            self.input_camera_names is not None and self.target_camera_names is not None
        )
        if use_input_target:
            data_lists["is_input"] = []
            # Create set for quick lookup of input cameras
            input_cam_set = set(self.input_camera_names)
        else:
            input_cam_set = None

        # Single loop over all cameras (input cameras first, then target cameras when configured)
        for cam_name in self._get_camera_names(log_id):
            view_identifier = {
                "log_id": log_id,
                "timestamp": center_timestamp,
                "cam_name": cam_name,
            }
            view_data = self._process_view(view_identifier)
            if view_data:
                for key, value in view_data.items():
                    data_lists[key].extend(value)
                # Mark is_input if input/target cameras are configured
                if use_input_target:
                    num_images_from_view = len(view_data["images"])
                    is_input = cam_name in input_cam_set
                    data_lists["is_input"].extend([is_input] * num_images_from_view)

        if len(data_lists["images"]) < self.min_views:
            logger.warning(
                f"Sample {log_id}/{center_timestamp} missing views. Found {len(data_lists['images'])}. Skipping."
            )
            return self.__getitem__((index + 1) % len(self))

        reference_rotation = self._get_reference_rotation(
            log_id, center_timestamp, data_lists["poses"]
        )
        reference_translation = self._get_reference_translation(
            log_id, center_timestamp, data_lists["poses"]
        )
        # --- Fetch per-sample satellite views ---
        if self.use_satellite:
            if self.dino_satellite_feats_dir is not None:
                data_lists["satellite_features"] = self._get_satellite_view_features(
                    log_id,
                    center_timestamp,
                )
                data_lists["satellite_features"] = self._project_satellite_features(
                    data_lists["satellite_features"]
                )
            data_lists["satellite_images"] = self._get_satellite_views(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )
        if self.use_osm:
            data_lists["osm_images"] = self._get_osm_views(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )
        if self.use_height_maps:
            data_lists["height_maps"] = self._get_height_maps(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )

        sample_id = (log_id, center_timestamp, 0)
        return self._package_data(data_lists, sample_id, reference_rotation, reference_translation)

    def _rotate_satellite_image(
        self,
        sat_img: torch.Tensor,
        reference_rotation: torch.Tensor,
        view_intrinsics: torch.Tensor,
        view_w_T_cam: torch.Tensor,
    ) -> torch.Tensor:
        """The reference rotation is that of a camera oriented in ENU, i.e Z along north."""
        view_ref_R_cam = reference_rotation.transpose(-1, -2) @ view_w_T_cam[:3, :3]
        # measure angle from X to Z of camera in reference frame
        z_rotation = torch.atan2(view_ref_R_cam[2, 2], view_ref_R_cam[0, 2]).item()
        # Compute the rotation from the pp to the image center
        x_angle = torch.atan(
            (view_intrinsics[0, 2] - self.resolution_width / 2) / view_intrinsics[0, 0]
        ).item()
        rotation_angle = 90 - np.rad2deg(z_rotation + x_angle)

        view_sat_img = F.rotate(sat_img, rotation_angle, interpolation=InterpolationMode.BILINEAR)

        return view_sat_img

    def _select_timestamps_for_sample(self, index: int) -> List[int]:
        log_id, center_timestamp = self.samples[index]
        all_timestamps = self._get_timestamps_for_log(log_id)

        center_idx = all_timestamps.index(center_timestamp)
        offset = (self.temporal_window_size - 1) // 2
        start_idx = max(0, center_idx - offset)
        end_idx = min(len(all_timestamps), start_idx + self.temporal_window_size)
        window_timestamps = all_timestamps[start_idx:end_idx]
        return window_timestamps

    def _get_temporal_item(self, index: int) -> Dict[str, Union[torch.Tensor, Tuple]]:
        log_id, center_timestamp = self.samples[index]
        data_lists = {"images": [], "poses": [], "intrinsics": [], "timestamps": []}

        # Track is_input flags if input/target cameras are configured
        use_input_target = (
            self.input_camera_names is not None and self.target_camera_names is not None
        )
        if use_input_target:
            data_lists["is_input"] = []
            # Create set for quick lookup of input cameras
            input_cam_set = set(self.input_camera_names)
        else:
            input_cam_set = None

        if self.mode in ("val", "test"):
            camera_names = list(self._get_camera_names(log_id))
            camera_names.sort()
            selected_cameras = camera_names[: self.num_subsample_cameras]
        else:
            num_subsample_cameras = random.randint(
                self.min_subsample_cameras, self.max_subsample_cameras
            )
            selected_cameras = random.sample(self._get_camera_names(log_id), num_subsample_cameras)

        try:
            window_timestamps = self._select_timestamps_for_sample(index)
        except ValueError as e:
            logger.warning(
                f"Center timestamp {center_timestamp} not found in log {log_id}. Skipping. Error: {e}"
            )
            return self.__getitem__((index + 1) % len(self))
        timestamp_to_window_index = {ts: idx for idx, ts in enumerate(window_timestamps)}

        candidate_pool = [
            {"log_id": log_id, "timestamp": ts, "cam_name": cam}
            for ts in window_timestamps
            for cam in selected_cameras
        ]

        num_views_to_sample = (
            self.num_max_images // 2 if self.cropping_method == "dual" else self.num_max_images
        )
        num_views_to_sample = min(num_views_to_sample, len(candidate_pool))
        if num_views_to_sample < len(candidate_pool):
            if self.mode in ("val", "test"):
                sampled_views = candidate_pool[:num_views_to_sample]
            else:
                sampled_views = random.sample(candidate_pool, num_views_to_sample)
        else:
            sampled_views = candidate_pool

        for view_identifier in sampled_views:
            view_data = self._process_view(view_identifier)
            if view_data:
                for key, value in view_data.items():
                    data_lists[key].extend(value)
                # Mark is_input if input/target cameras are configured
                if use_input_target:
                    num_images_from_view = len(view_data["images"])
                    is_input = view_identifier["cam_name"] in input_cam_set
                    if is_input and self.temporal_window_input_mask is not None:
                        view_timestamp = view_identifier["timestamp"]
                        ts_window_idx = timestamp_to_window_index.get(view_timestamp, None)
                        if ts_window_idx is None or ts_window_idx >= len(self.temporal_window_input_mask):
                            raise ValueError(
                                "Could not map view timestamp to temporal window index for "
                                f"timestamp={view_timestamp} in sample {(log_id, center_timestamp)}."
                            )
                        is_input = self.temporal_window_input_mask[ts_window_idx]
                    data_lists["is_input"].extend([is_input] * num_images_from_view)
            else:
                logger.warning(f"View not found: {view_identifier}")

        if len(data_lists["images"]) < self.min_views:
            logger.warning(
                f"Temporal sample {log_id}/{center_timestamp} could not gather enough views. Skipping."
            )
            return self.__getitem__((index + 1) % len(self))

        reference_rotation = self._get_reference_rotation(
            log_id, center_timestamp, data_lists["poses"]
        )
        reference_translation = self._get_reference_translation(
            log_id, center_timestamp, data_lists["poses"]
        )

        # --- Fetch per-sample satellite views ---
        if self.use_satellite:
            if self.dino_satellite_feats_dir is not None:
                data_lists["satellite_features"] = self._get_satellite_view_features(
                    log_id,
                    center_timestamp,
                )
                data_lists["satellite_features"] = self._project_satellite_features(
                    data_lists["satellite_features"]
                )
            data_lists["satellite_images"] = self._get_satellite_views(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )

        if self.use_osm:
            data_lists["osm_images"] = self._get_osm_views(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )
        if self.use_height_maps:
            data_lists["height_maps"] = self._get_height_maps(
                log_id,
                center_timestamp,
                reference_rotation,
                data_lists["poses"],
                data_lists["intrinsics"],
            )

        duration = window_timestamps[-1] - window_timestamps[0] if len(window_timestamps) > 1 else 0
        sample_id = (log_id, center_timestamp, duration)
        return self._package_data(data_lists, sample_id, reference_rotation, reference_translation)
