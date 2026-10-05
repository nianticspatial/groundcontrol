import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from metropolis import Metropolis
from metropolis.utils.geo import TopocentricConverter
from omegaconf import ListConfig
from PIL import Image
from pyquaternion import Quaternion

from osm_diffusion.data.generic_mvs_dataset import GenericMVSDataset

logger = logging.getLogger(__name__)


class MetropolisDataset(GenericMVSDataset):
    """
    Concrete implementation for the Metropolis Dataset, inheriting from GenericMVSDataset.
    This version assumes source images are square and only performs scaling.
    """

    @staticmethod
    def resize_square_image(
        image: Image.Image, intrinsics: np.ndarray, resolution: Union[int, Tuple[int, int]]
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Resizes an image to a target resolution and scales intrinsics accordingly.
        Supports both square and non-square target resolutions.
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
        if original_width == target_width and original_height == target_height:
            return image, intrinsics

        # Resize the image
        resized_image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

        # Scale the intrinsics matrix
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        updated_intrinsics = intrinsics.copy()
        updated_intrinsics[0, :] *= scale_x
        updated_intrinsics[1, :] *= scale_y

        return resized_image, updated_intrinsics

    # --- The _build_samples, _get_camera_names, and other methods remain the same ---

    def _build_samples(self):
        """
        Builds the list of samples for the Metropolis dataset. It caches scene-to-timestamp
        and timestamp-to-sample-token mappings for efficient data retrieval.
        """
        data_dir = Path(self.config["data_dir"])
        self.metropolis = Metropolis(self.mode, dataroot=str(data_dir))

        # Initialize caches for temporal sampling and efficient lookups
        self.scene_to_timestamps = {}
        self.timestamp_to_sample_token = {}

        # The TopocentricConverter is needed for GPS coordinate conversion
        geo_ref = self.metropolis.geo["reference"]
        self.converter = TopocentricConverter(geo_ref["lat"], geo_ref["lon"], geo_ref["alt"])

        sampling_interval = self.config.get("sampling_interval", 10)
        start_offset = self.config.get("start_offset_timestamps", 5)
        window_sampling_interval = self.config.get("temporal_window_sampling_interval", 1)
        window_start_offset = self.config.get("temporal_window_start_offset", 1)
        single_sample_per_scene = self.config.get("single_sample_per_scene", False)

        self._camera_names = ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT"]

        for scene in self.metropolis.scene:
            scene_token = scene["token"]
            current_sample_token = scene["first_sample_token"]
            scene_timestamps = []

            while current_sample_token:
                sample = self.metropolis.get("sample", current_sample_token)
                # We only consider samples that have front camera data
                if "CAM_FRONT" in sample["data"]:
                    timestamp = sample["timestamp"]
                    scene_timestamps.append(timestamp)
                    self.timestamp_to_sample_token[timestamp] = sample["token"]

                current_sample_token = sample["next_sample"]

            if not scene_timestamps:
                continue

            # Sort timestamps before subsampling
            scene_timestamps = sorted(scene_timestamps)

            # Apply window sampling if configured
            if window_sampling_interval > 1 or window_start_offset > 0:
                subsampled_timestamps = []
                for i, ts in enumerate(scene_timestamps):
                    if (i - window_start_offset) >= 0 and (
                        i - window_start_offset
                    ) % window_sampling_interval == 0:
                        subsampled_timestamps.append(ts)
            else:
                subsampled_timestamps = scene_timestamps

            self.scene_to_timestamps[scene_token] = subsampled_timestamps

            # Populate self.samples with (scene_token, timestamp) tuples
            this_scene_samples = []
            for i, ts in enumerate(subsampled_timestamps):
                if (i - start_offset) >= 0 and (i - start_offset) % sampling_interval == 0:
                    this_scene_samples.append((scene_token, ts))
                if single_sample_per_scene and len(this_scene_samples) == 1:
                    break
            self.samples.extend(this_scene_samples)

    def _get_camera_names(self, log_id) -> List[str]:
        """Returns the list of camera identifiers for Metropolis."""
        # If input/target cameras are specified, return combined list
        # This ensures _get_camera_names() returns the correct order when used elsewhere
        if hasattr(self, "input_camera_names") and hasattr(self, "target_camera_names"):
            if self.input_camera_names is not None and self.target_camera_names is not None:
                # Input cameras first, then target cameras
                return self.input_camera_names + self.target_camera_names
        return self._camera_names

    def _get_timestamps_for_log(self, log_id: str) -> List[int]:
        """For Metropolis, 'log_id' is the scene_token."""
        return self.scene_to_timestamps.get(log_id, [])

    def _process_view(self, view_identifier: Dict) -> Optional[Dict[str, List]]:
        """
        Processes a single camera view. This version only scales the image,
        assuming the source is already square.
        """
        log_id = view_identifier["log_id"]
        timestamp = view_identifier["timestamp"]
        cam_name = view_identifier["cam_name"]

        sample_token = self.timestamp_to_sample_token.get(timestamp)
        if not sample_token:
            logger.warning(f"Could not find sample_token for timestamp {timestamp}")
            return None

        sample = self.metropolis.get("sample", sample_token)
        if cam_name not in sample["data"]:
            return None

        sample_data_token = sample["data"][cam_name]
        sd_record = self.metropolis.get("sample_data", sample_data_token)
        image_path = Path(self.metropolis.get_sample_data_path(sample_data_token))

        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")
        cs_record = self.metropolis.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        original_intrinsics = np.array(cs_record["camera_intrinsic"])

        ep_record = self.metropolis.get("ego_pose", sd_record["ego_pose_token"])
        cam_timestamp_s = torch.tensor(ep_record["timestamp"], dtype=torch.float64) / 1e6

        ego_to_world = Quaternion(ep_record["rotation"]).transformation_matrix
        ego_to_world[:3, 3] = ep_record["translation"]
        cam_to_ego = Quaternion(cs_record["rotation"]).transformation_matrix
        cam_to_ego[:3, 3] = cs_record["translation"]
        pose = torch.from_numpy(ego_to_world @ cam_to_ego).float()

        processed_img, K = self.resize_square_image(image, original_intrinsics, self.resolution)

        return {
            "images": [self.transform(processed_img)],
            "poses": [pose],
            "intrinsics": [torch.from_numpy(K).float()],
            "timestamps": [cam_timestamp_s],
        }

    def _get_pose(self, view_identifier: Dict) -> Optional[torch.Tensor]:
        timestamp = view_identifier["timestamp"]
        cam_name = view_identifier["cam_name"]

        sample_token = self.timestamp_to_sample_token.get(timestamp)
        if not sample_token:
            logger.warning(f"Could not find sample_token for timestamp {timestamp}")
            return None

        sample = self.metropolis.get("sample", sample_token)
        if cam_name not in sample["data"]:
            return None

        sample_data_token = sample["data"][cam_name]
        sd_record = self.metropolis.get("sample_data", sample_data_token)

        ep_record = self.metropolis.get("ego_pose", sd_record["ego_pose_token"])
        ego_to_world = Quaternion(ep_record["rotation"]).transformation_matrix
        ego_to_world[:3, 3] = ep_record["translation"]
        cs_record = self.metropolis.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        cam_to_ego = Quaternion(cs_record["rotation"]).transformation_matrix
        cam_to_ego[:3, 3] = cs_record["translation"]
        pose = torch.from_numpy(ego_to_world @ cam_to_ego).float()
        return pose

    def _get_reference_rotation(
        self, log_id: str, timestamp: Any, poses: list[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Returns a fixed ENU (East-North-Up) aligned reference rotation.
        This ensures consistency for aligning satellite imagery across datasets.
        The reference frame is: X=East, Y=Down, Z=North.
        """
        # A fixed rotation matrix: Z along North (city Y), X along East (city X), Y along Down (city -Z)
        city_SO3_reference = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        return city_SO3_reference

    def _get_reference_translation(
        self, log_id: str, timestamp: Any, poses: List[torch.Tensor]
    ) -> torch.Tensor:
        sample_token = self.timestamp_to_sample_token.get(timestamp)
        sample = self.metropolis.get("sample", sample_token)

        sample_data_token = sample["data"]["CAM_FRONT"]
        sd_record = self.metropolis.get("sample_data", sample_data_token)
        ep_record = self.metropolis.get("ego_pose", sd_record["ego_pose_token"])
        return torch.tensor(ep_record["translation"]).float()

    def _poses_to_gps(self, poses: torch.Tensor, log_id: str) -> torch.Tensor:
        """
        Converts a batch of camera poses from topocentric world coordinates to WGS84 GPS coordinates.
        """
        world_coords = poses[:, :3, 3].cpu().numpy()
        lats, lons, _ = self.converter.to_lla(
            world_coords[:, 0], world_coords[:, 1], world_coords[:, 2]
        )
        gps_coords = np.stack([lats, lons], axis=-1)
        return torch.from_numpy(gps_coords).float()

    def _get_osm_views(
        self,
        log_id: str,
        timestamp: Any,
        reference_rotation: Optional[torch.Tensor],
        views_w_T_cam: Optional[List[torch.Tensor]],
        intrinsics: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """(Abstract) Loads all configured OSM views for a given sample."""
        return None

    def _get_satellite_views(
        self,
        log_id: str,  # This is the scene_token
        timestamp: Any,
        reference_rotation: Optional[torch.Tensor],
        views_w_T_cam: Optional[List[torch.Tensor]],
        intrinsics: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        """
        Loads pre-processed satellite views from disk for a given sample.
        """
        if not self.use_satellite:
            return None

        satellite_images = []
        for sat_config in self.satellite_configs:
            # Construct the path to the pre-processed image based on the
            # output structure of the preprocessing script.
            sat_path = (
                self.satellite_dir / self.mode / log_id / sat_config["name"] / f"{timestamp}.png"
            )
            resolution = sat_config["resolution"]

            if sat_path.exists():
                sat_img_pil = Image.open(sat_path).convert("RGB")
                sat_img = self.transform(sat_img_pil)

                if sat_config.get("rotate_per_view", False) and views_w_T_cam:
                    for view_w_T_cam, view_intrinsics in zip(views_w_T_cam, intrinsics):
                        view_sat_img = self._rotate_satellite_image(
                            sat_img, reference_rotation, view_intrinsics, view_w_T_cam
                        )
                        satellite_images.append(view_sat_img)
                else:
                    satellite_images.append(F.resize(sat_img, (resolution, resolution)))
            else:
                logger.warning(
                    f"Missing pre-processed satellite image at {sat_path}, appending zeros."
                )
                # Keep per-view cardinality even when the satellite tile is missing.
                if sat_config.get("rotate_per_view", False) and views_w_T_cam:
                    for _ in views_w_T_cam:
                        satellite_images.append(torch.zeros(3, resolution, resolution))
                else:
                    # Create a single placeholder tensor if we are not rotating per view.
                    satellite_images.append(torch.zeros(3, resolution, resolution))

        return satellite_images

    def _get_satellite_view_features(
        self,
        log_id: str,
        timestamp: Any,
    ) -> Optional[List[torch.Tensor]]:
        dino_features_list = []
        if self.dino_satellite_feats_dir is None:
            return dino_features_list

        for sat_config in self.satellite_configs:
            feat_path = (
                Path(self.dino_satellite_feats_dir)
                / self.mode
                / log_id
                / sat_config["name"]
                / f"{timestamp}.npz"
            )
            if feat_path.exists():
                dino_features = torch.from_numpy(np.load(feat_path)["feats"])
                dino_features = dino_features.reshape(32, 32, 1024)
                dino_features_list.append(dino_features)
            else:
                feat_path = (
                    Path(self.dino_satellite_feats_dir) / self.mode / log_id / f"{timestamp}.npz"
                )
                if feat_path.exists():
                    dino_features = torch.from_numpy(np.load(feat_path)["feats"])
                    dino_features = dino_features.reshape(32, 32, 1024)
                    dino_features_list.append(dino_features)
                else:
                    sat_img_path = (
                        self.satellite_dir
                        / self.mode
                        / log_id
                        / sat_config["name"]
                        / f"{timestamp}.png"
                    )
                    if sat_img_path.exists():
                        raise FileNotFoundError(
                            f"satellite image exists but DINO features do not for {sat_img_path}"
                        )
                    print(f"DINO features not found for {sat_img_path}, using zeros")
                    dino_features_list.append(torch.zeros(32, 32, 1024, dtype=torch.float32))
        return dino_features_list
