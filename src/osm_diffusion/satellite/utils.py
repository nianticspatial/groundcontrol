import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2

from osm_diffusion.satellite.dpt_sat_encoder import SatelliteDPTEncoder
from osm_diffusion.satellite.osm_classes import CLASS_MAPPING

os.environ.setdefault("TORCH_HOME", "/mnt/nas/personal/akshay/model_cache/torch_hub")
os.environ.setdefault("HF_HOME", "/mnt/nas/personal/akshay/model_cache/hf_hub")


def one_hot_encode_raster(raster: np.ndarray) -> np.ndarray:
    """
    Converts a 3-channel raster into a one-hot encoded representation.

    The input raster has 3 channels for points, lines, and areas. Each pixel
    in a channel contains an integer `idx + 1` corresponding to a class,
    or 0 for no class.

    This function creates a one-hot vector for each category and concatenates
    them into a single multi-channel raster.

    Args:
        raster (np.ndarray): A 3D NumPy array of shape (H, W, 3) where the
                             channels represent points, lines, and areas respectively.
                             Values are integers representing class indices + 1.

    Returns:
        np.ndarray: A 3D NumPy array of shape (H, W, N) where N is the total
                    number of classes across all categories. The array contains
                    the concatenated one-hot encodings.
    """
    if not isinstance(raster, np.ndarray) or raster.ndim != 3 or raster.shape[2] != 3:
        raise ValueError("Input raster must be a NumPy array with shape (H, W, 3).")

    # Get the number of classes for each category from the mapping
    num_points = len(CLASS_MAPPING["points"])
    num_lines = len(CLASS_MAPPING["lines"])
    num_areas = len(CLASS_MAPPING["areas"])

    # Extract individual channels from the input raster
    points_channel = raster[..., 2]
    lines_channel = raster[..., 1]
    areas_channel = raster[..., 0]

    points_one_hot = (np.arange(num_points) == points_channel[..., None] - 1).astype(np.uint8)
    lines_one_hot = (np.arange(num_lines) == lines_channel[..., None] - 1).astype(np.uint8)
    areas_one_hot = (np.arange(num_areas) == areas_channel[..., None] - 1).astype(np.uint8)

    # Concatenate the one-hot encoded arrays along the last axis (the class dimension)
    one_hot_raster = np.concatenate([points_one_hot, lines_one_hot, areas_one_hot], axis=-1)

    return one_hot_raster


def get_sat_encoder(
    sat_encoder_name, use_dpt_head: bool = False, dpt_head_config: dict | None = None
):
    sat_backbone, bb_size = sat_encoder_name.split("_")[:2]
    if sat_backbone == "dinov2":
        sat_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{bb_size}_reg_lc").backbone
        del sat_encoder.mask_token
    elif sat_backbone == "dinov3" or sat_encoder_name.startswith("dinov3"):
        sat_encoder = get_dinov3_model(sat_encoder_name)
    if use_dpt_head:
        sat_encoder = SatelliteDPTEncoder(sat_encoder, dpt_head_config=dpt_head_config)
    return sat_encoder


def get_sat_encoder_transform(sat_encoder_name):
    if sat_encoder_name.startswith("dinov2"):
        return transforms.Compose(
            [
                lambda x: 255.0 * (x * 0.5 + 0.5),  # need to undo the dataset transform
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ]
        )
    elif sat_encoder_name.startswith("dinov3"):
        return transforms.Compose(
            [
                lambda x: x * 0.5 + 0.5,  # need to undo the dataset transform
                transforms.Normalize(
                    mean=(0.430, 0.411, 0.296),
                    std=(0.213, 0.156, 0.143),
                ),
            ]
        )
    else:
        raise ValueError("Unsupported satellite encoder")


def get_dinov3_model(model_name):
    """Load a local DINOv3 backbone. Not used when satellite features are precomputed."""
    if not model_name.startswith("dinov3"):
        raise ValueError(f"Unsupported model name: {model_name}")
    dinov3_dir = os.environ.get("DINOV3_REPO")
    weights = os.environ.get("DINOV3_WEIGHTS")
    if not dinov3_dir or not weights:
        raise RuntimeError(
            "On-the-fly DINOv3 features need DINOV3_REPO (local torch.hub checkout) "
            "and DINOV3_WEIGHTS (checkpoint file). Metropolis inference uses "
            "precomputed features and does not load this encoder."
        )
    return torch.hub.load(dinov3_dir, "dinov3_vitl16", source="local", weights=weights)


# def sample_ray_points(image_width, intrinsics, latent_ds, )       # Pick a row of the image, and sample points along a ray for each pixel.
#     # B, N, C, H, W = batch["images"].shape
#     latents_W = W // self.vae_downsampling_factor
#     u = torch.arange(latents_W, device=self.device)[None, None]
#     intrinsics = batch["intrinsics"].clone()
#     intrinsics = intrinsics.to(self.device)
#     intrinsics_scale = 1.0 / self.vae_downsampling_factor
#     intrinsics[..., :2, :3] *= intrinsics_scale
#     x = (u - intrinsics[:, :, 0, 2, None]) / intrinsics[:, :, 0, 0, None]  # B, N, latents_W
#     x = x.unsqueeze(-1).repeat(1, 1, 1, num_points)  # B, N, latents_W, num_points
#     z_vals = torch.linspace(
#         self._sample_points_min_depth_px,
#         self._sample_points_max_depth_px,
#         num_points,
#         device=self.device,
#     )
#     z_vals = z_vals[None, None, None, :].repeat(
#         B, N, latents_W, 1
#     )  # B, N, latents_W, num_points
#     ray_points = (
#         torch.stack([x, -torch.ones_like(x)], dim=-1) * z_vals[..., None]
#     )  # B, N, latents_W, num_points, 2
#     return ray_points


def compute_yaw_aligned_ray_directions(batch, latent_H, latent_W, vae_downsampling_factor, device):
    """
    Computes the 3D ray direction vectors for each latent pixel, expressed in a
    yaw-aligned reference frame. This version correctly defines "yaw" relative
    to the image center, not the principal point, to match the semantic conditioning.

    Args:
        batch (dict): The input batch containing images, intrinsics, camera_poses,
                      and reference_rotation.
        vae_downsampling_factor (int): The factor by which the VAE downsamples the image.
        device: The torch device to perform computations on.

    Returns:
        torch.Tensor: The final ray directions. Shape: (B, N, H, W, 3)
    """
    # --- 1. Get initial ray directions in local camera coordinates (Unchanged) ---
    j, i = torch.meshgrid(
        torch.arange(latent_H, device=device, dtype=torch.float32),
        torch.arange(latent_W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    intrinsics = batch["intrinsics"].to(device)
    intrinsics_latent = intrinsics.clone()
    intrinsics_latent[..., :2, :3] *= 1.0 / vae_downsampling_factor
    intrinsics_latent = intrinsics_latent[:, :, None, None]

    x = (i.unsqueeze(0).unsqueeze(0) - intrinsics_latent[..., 0, 2]) / intrinsics_latent[..., 0, 0]
    y = (j.unsqueeze(0).unsqueeze(0) - intrinsics_latent[..., 1, 2]) / intrinsics_latent[..., 1, 1]
    z = torch.ones_like(x)

    initial_dirs = torch.stack([x, y, z], dim=-1)
    initial_dirs = F.normalize(initial_dirs, p=2, dim=-1)

    # --- 2. Calculate the full rotation from camera to reference frame (Unchanged) ---
    cam_poses = batch["camera_poses"].to(device)
    ref_rot = batch["reference_rotation"].to(device)
    R_ref_from_cam = ref_rot[:, None].transpose(-1, -2) @ cam_poses[..., :3, :3]

    # (3a) Calculate the yaw of the camera's principal axis (z_rot_clock)
    cam_z_in_ref = R_ref_from_cam[..., :, 2]
    yaw_of_principal_axis = torch.atan2(cam_z_in_ref[..., 0], cam_z_in_ref[..., 2])

    # (3b) Calculate the angular offset from the principal axis to the image center ray (x_angle)
    # This uses the original, full-resolution intrinsics and image width.
    img_center = latent_W * vae_downsampling_factor / 2.0
    cx = intrinsics[..., 0, 2]  # Principal point x
    fx = intrinsics[..., 0, 0]  # Focal length x
    principal_point_offset_angle = torch.atan2(cx - img_center, fx)

    # (3c) Calculate the final yaw of the image center (img_center_rot_clock)
    yaw_of_image_center = yaw_of_principal_axis - principal_point_offset_angle

    # (3d) Build the 3x3 yaw rotation matrix using the corrected angle
    c, s = torch.cos(yaw_of_image_center), torch.sin(yaw_of_image_center)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)

    R_yaw_only = torch.stack(
        [
            torch.stack([c, zeros, s], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-s, zeros, c], dim=-1),
        ],
        dim=-2,
    )

    # --- 4. Isolate the pitch and roll rotation (Unchanged) ---
    R_pitch_roll = R_yaw_only.transpose(-1, -2) @ R_ref_from_cam

    # --- 5. Apply the rotation to the initial rays (Unchanged) ---
    final_dirs = (R_pitch_roll.unsqueeze(-3).unsqueeze(-3) @ initial_dirs.unsqueeze(-1)).squeeze(-1)

    return final_dirs


class RayPositionalEncoder(nn.Module):
    """
    Vectorized sine-cosine positional encoder for input points.

    Args:
        d_input (int): Number of input dimensions (e.g., 3 for 3D coordinates).
        n_freqs (int): Number of frequency bands (L in the paper).
        log_space (bool): If True, frequencies are sampled in log scale.
        include_input (bool): If True, concatenates the original input to the encoding.
    """

    def __init__(
        self, d_input: int, n_freqs: int, log_space: bool = True, include_input: bool = True
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.include_input = include_input

        # Calculate output dimension
        self.d_output = self.d_input * (2 * self.n_freqs)
        if self.include_input:
            self.d_output += self.d_input

        # Define frequency bands
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (self.n_freqs - 1), self.n_freqs)

        # Register frequency bands as a buffer. This is not a trainable parameter,
        # but it should be moved to the correct device (e.g., GPU) with the model.
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x):
        """
        Apply positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [..., d_input].

        Returns:
            torch.Tensor: Encoded tensor of shape [..., d_output].
        """
        # 1. Repeat the input tensor for each frequency band.
        # x shape: [..., d_input]
        # x_expanded shape: [..., d_input, n_freqs]
        # This is achieved by broadcasting x [..., d_input, 1] with freq_bands [n_freqs]
        x_expanded = x.unsqueeze(-1) * self.freq_bands

        # 2. Apply sine and cosine functions.
        # sincos shape: [..., d_input, 2 * n_freqs]
        # We concatenate the sin and cos outputs along the last dimension.
        sincos = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)

        # 3. Reshape to a flat vector.
        # embedding shape: [..., d_input * 2 * n_freqs]
        # This flattens the last two dimensions into one.
        embedding = sincos.reshape(*sincos.shape[:-2], -1)

        # 4. (Optional) Concatenate the original input.
        if self.include_input:
            embedding = torch.cat([x, embedding], dim=-1)

        return embedding


def one_hot_encode_heightmaps(height_map, max_height=200, num_bins=200):
    """
    One-hot encodes height map values into discrete bins.

    Args:
        height_map (torch.Tensor): Input height map tensor of shape (B, 1, H, W).
        min_height (float): Minimum height value for binning.
        max_height (float): Maximum height value for binning.
        num_bins (int): Number of discrete bins.
    Returns:
        torch.Tensor: One-hot encoded height map of shape (B, num_bins, H, W).
    """
    B, N, C, H, W = height_map.shape
    height_map = height_map.view(B * N, C, H, W)
    assert C == 1, "Height map should have a single channel."

    valid_mask = torch.isfinite(height_map) & (~torch.isnan(height_map))  # B * N, 1, H, W
    min_height = (height_map + (valid_mask * 1e6)).reshape(B * N, -1).min(dim=-1).values
    height_map = height_map - min_height[:, None, None, None]

    # 1. Normalize height values to [0, 1]
    height_map = torch.clamp(height_map, max=max_height, min=0.0)
    height_map[~valid_mask] = 0.0
    height_map_normalized = height_map / max_height

    # 2. Scale to [0, num_bins - 1] and convert to integer bin indices
    height_map_scaled = height_map_normalized * (num_bins - 1)
    bin_indices = torch.round(height_map_scaled).long()

    # 3. One-hot encode the bin indices
    one_hot_height_map = F.one_hot(bin_indices, num_classes=num_bins)

    # 4. Reshape to (B, H, W, num_bins)
    one_hot_height_map = one_hot_height_map.float()  # B * N, 1, H, W, num_bins
    one_hot_height_map = one_hot_height_map * valid_mask[..., None].float()

    one_hot_height_map = one_hot_height_map.view(B, N, H, W, num_bins)

    return one_hot_height_map
