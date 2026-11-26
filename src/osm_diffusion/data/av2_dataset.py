import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import av2.geometry.utm as utm_utils
import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from osm_diffusion.data.generic_mvs_dataset import GenericMVSDataset

logger = logging.getLogger(__name__)


class ArgoverseDataset(GenericMVSDataset):
    """Concrete implementation for the Argoverse 2 Sensor Dataset."""

    def __init__(self, config: Dict, mode: str = "train"):
        super().__init__(config, mode)
        # Convert input/target camera names from strings to RingCameras enums if provided
        if self.input_camera_names is not None:
            self.input_camera_names = [
                cam if isinstance(cam, RingCameras) else RingCameras(cam)
                for cam in self.input_camera_names
            ]
        if self.target_camera_names is not None:
            self.target_camera_names = [
                cam if isinstance(cam, RingCameras) else RingCameras(cam)
                for cam in self.target_camera_names
            ]

    def _build_samples(self):
        """Builds the list of samples and caches timestamps for Argoverse 2."""
        data_dir = Path(self.config["data_dir"]) / self.mode
        # This loader is specific to Argoverse
        self.loader = AV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
        self.log_to_timestamps = {}

        sampling_interval = self.config.get("sampling_interval", 10)
        start_offset = self.config.get("start_offset_timestamps", 5)

        for log_id in self.loader.get_log_ids():
            timestamps = self.loader.get_ordered_log_lidar_timestamps(log_id)
            self.log_to_timestamps[log_id] = timestamps
            for i, ts in enumerate(timestamps):
                if (i - start_offset) >= 0 and (i - start_offset) % sampling_interval == 0:
                    if self.use_satellite:
                        satellite_available = False
                        for sat_config in self.satellite_configs:
                            sat_path = (
                                self.satellite_dir
                                / self.mode
                                / log_id
                                / sat_config["name"]
                                / f"{ts}.png"
                            )
                            if sat_path.exists():
                                satellite_available = True
                                break
                        if not satellite_available:
                            print(
                                f"Missing satellite image for log {log_id} at timestamp {ts}, skipping sample."
                            )
                            continue
                    self.samples.append((log_id, ts))

    def _get_camera_names(self, log_id) -> List[RingCameras]:
        # If input/target cameras are specified, return combined list
        # This ensures _get_camera_names() returns the correct order when used elsewhere
        if hasattr(self, "input_camera_names") and hasattr(self, "target_camera_names"):
            if self.input_camera_names is not None and self.target_camera_names is not None:
                # Input cameras first, then target cameras
                # Camera names are already converted to RingCameras enums in __init__
                return self.input_camera_names + self.target_camera_names
        return list(RingCameras)

    def _get_timestamps_for_log(self, log_id: str) -> List[int]:
        return self.log_to_timestamps.get(log_id, [])

    def _get_satellite_views(
        self,
        log_id: str,
        timestamp_ns: int,
        reference_rotation: Optional[torch.Tensor] = None,
        views_w_T_cam: Optional[List[torch.Tensor]] = None,
        intrinsics: Optional[List[torch.Tensor]] = None,
    ) -> Optional[List[torch.Tensor]]:
        """
        Loads all configured satellite views for the given log and timestamp.
        """
        if not self.use_satellite:
            return None

        satellite_images = []
        for sat_config in self.satellite_configs:
            size, ppm, resolution = sat_config["size"], sat_config["ppm"], sat_config["resolution"]
            sat_path = (
                self.satellite_dir / self.mode / log_id / sat_config["name"] / f"{timestamp_ns}.png"
            )

            if sat_path.exists():
                try:
                    sat_img = Image.open(sat_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading satellite image at {sat_path}: {e}")
                    logger.warning(f"Error loading satellite image at {sat_path}: {e}")
                    satellite_images.append(torch.zeros(3, resolution, resolution))
                    continue
                sat_img = self.transform(sat_img)
                if sat_config["rotate_per_view"]:
                    for view_w_T_cam, view_intrinsics in zip(views_w_T_cam, intrinsics):
                        view_sat_img = self._rotate_satellite_image(
                            sat_img, reference_rotation, view_intrinsics, view_w_T_cam
                        )
                        # center crop after rotation
                        crop_size = int(size * ppm / math.sqrt(2))
                        view_sat_img = F.center_crop(view_sat_img, (crop_size, crop_size))
                        view_sat_img = F.resize(view_sat_img, (resolution, resolution))
                        satellite_images.append(view_sat_img)
                else:
                    if sat_config["rotate"]:
                        x_rotation = torch.atan2(
                            reference_rotation[1, 0], reference_rotation[0, 0]
                        ).item()
                        rotation_angle = 90 - np.rad2deg(x_rotation)
                        sat_img = F.rotate(
                            sat_img, rotation_angle, interpolation=InterpolationMode.BILINEAR
                        )
                        # center crop after rotation
                        crop_size = int(size * ppm / math.sqrt(2))
                        sat_img = F.center_crop(sat_img, (crop_size, crop_size))
                    sat_img = F.resize(sat_img, (resolution, resolution))
                    satellite_images.append(sat_img)
            else:
                logger.warning(f"Missing satellite image at {sat_path}, appending zeros.")
                # The size of the dummy tensor should match the expected output size
                satellite_images.append(torch.zeros(3, resolution, resolution))

        return satellite_images

    def _process_view(self, view_identifier: Dict) -> Optional[Dict[str, List]]:
        log_id = view_identifier["log_id"]
        timestamp_ns = view_identifier["timestamp"]
        cam_enum = view_identifier["cam_name"]

        cam_name = cam_enum.value
        img_fpath = self.loader.get_closest_img_fpath(log_id, cam_name, timestamp_ns)
        if not img_fpath or not img_fpath.exists():
            return None

        view_images, view_poses, view_intrinsics, view_timestamps = [], [], [], []
        image = Image.open(img_fpath).convert("RGB")
        pinhole_cam = self.loader.get_log_pinhole_camera(log_id, cam_name)
        original_intrinsics = pinhole_cam.intrinsics.K

        cam_timestamp_ns = int(img_fpath.stem)
        cam_timestamp_ms = torch.tensor(cam_timestamp_ns).to(torch.float64) / 1e6
        city_SE3_ego = self.loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
        ego_SE3_cam = pinhole_cam.ego_SE3_cam
        city_SE3_cam = city_SE3_ego.compose(ego_SE3_cam)
        pose = torch.from_numpy(city_SE3_cam.transform_matrix).float()

        if self.cropping_method == "dual":
            img1, K1, img2, K2 = self.create_two_square_images(
                image, original_intrinsics, self.resolution
            )
            view_images.extend([self.transform(img1), self.transform(img2)])
            view_poses.extend([pose, pose])
            view_intrinsics.extend([torch.from_numpy(K1).float(), torch.from_numpy(K2).float()])
            view_timestamps.extend([cam_timestamp_ms, cam_timestamp_ms])
        else:
            processed_img, K = self.resize_and_crop(image, original_intrinsics, self.resolution)
            view_images.append(self.transform(processed_img))
            view_poses.append(pose)
            view_intrinsics.append(torch.from_numpy(K).float())
            view_timestamps.append(cam_timestamp_ms)

        return {
            "images": view_images,
            "poses": view_poses,
            "intrinsics": view_intrinsics,
            "timestamps": view_timestamps,
        }

    def _poses_to_gps(self, poses: torch.Tensor, log_id: str) -> torch.Tensor:
        """Converts camera poses to GPS coordinates using Argoverse utilities."""
        city_name = self.loader.get_city_name(log_id)
        city_coords = poses[:, :3, 3].cpu().numpy()
        gps_coords = utm_utils.convert_city_coords_to_wgs84(city_coords[:, :2], city_name)
        return torch.from_numpy(gps_coords).float()

    def _get_reference_rotation(
        self, log_id: str, timestamp_ns: int, poses: list[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """East-north-up reference: Z north, X east, Y down."""
        return torch.from_numpy(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])).float()

    def _get_reference_translation(
        self, log_id: str, timestamp_ns: int, poses: list[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Returns the 3x1 translation vector of the ego-vehicle itself as the reference.
        """
        city_SE3_ego = self.loader.get_city_SE3_ego(log_id, timestamp_ns)
        translation_vector = city_SE3_ego.transform_matrix[:3, 3]
        return torch.from_numpy(translation_vector).float()
