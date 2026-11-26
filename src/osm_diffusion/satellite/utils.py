import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


def get_sat_encoder(sat_encoder_name):
    sat_backbone, bb_size = sat_encoder_name.split("_")[:2]
    if sat_backbone == "dinov2":
        sat_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{bb_size}_reg_lc").backbone
        del sat_encoder.mask_token
        return sat_encoder
    if sat_backbone == "dinov3" or sat_encoder_name.startswith("dinov3"):
        return get_dinov3_model(sat_encoder_name)
    raise ValueError(f"Unsupported satellite encoder: {sat_encoder_name}")


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
    """Load a local DINOv3 backbone for satellite image encoding."""
    if not model_name.startswith("dinov3"):
        raise ValueError(f"Unsupported model name: {model_name}")
    dinov3_dir = os.environ.get("DINOV3_REPO")
    weights = os.environ.get("DINOV3_WEIGHTS")
    if not dinov3_dir or not weights:
        raise RuntimeError(
            "DINOv3 encoding requires DINOV3_REPO (local torch.hub checkout) "
            "and DINOV3_WEIGHTS (checkpoint file)."
        )
    return torch.hub.load(dinov3_dir, "dinov3_vitl16", source="local", weights=weights)


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
    # --- 1. Get initial ray directions in local camera coordinates ---
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

    # --- 2. Calculate the full rotation from camera to reference frame ---
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

    # --- 4. Isolate the pitch and roll rotation ---
    R_pitch_roll = R_yaw_only.transpose(-1, -2) @ R_ref_from_cam

    # --- 5. Apply the rotation to the initial rays ---
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
        x_expanded = x.unsqueeze(-1) * self.freq_bands
        sincos = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
        embedding = sincos.reshape(*sincos.shape[:-2], -1)
        if self.include_input:
            embedding = torch.cat([x, embedding], dim=-1)

        return embedding
