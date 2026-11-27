import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms

import groundcontrol.satellite.viz as viz_utils
from groundcontrol.satellite import utils as sat_utils
from groundcontrol.satellite.satellite_seva_pipeline import SatelliteSevaPipeline
from groundcontrol.seva.modules.transformer import PositionalEncoding


def get_satellite_feature_dim(sat_encoder_name: str) -> int:
    if "vitl" in sat_encoder_name:
        return 1024
    else:
        raise ValueError(f"Unknown satellite encoder {sat_encoder_name}")


class SatelliteRayGuidedCAPipeline(SatelliteSevaPipeline):
    def __init__(
        self,
        disable_sat_conditioning: bool = False,
        strict_dual_cfg_sat_mask: bool = False,
        train_sat_encoder: bool = False,
        train_sat_dpt_head: bool = False,
        sample_points_min_depth: float = 2,
        sample_points_max_depth: float = 100,
        num_sample_points: int = 16,
        sat_encoder_name: str = "dinov2_vits14",
        sat_use_dpt_head: bool = False,
        sat_dpt_head_config: dict | None = None,
        dinov3_repo: str | None = None,
        dinov3_weights: str | None = None,
        ray_pos_enc_freqs: int = 10,
        add_positional_encoding: bool = True,
        position_encoding_scale: float = 1.0,
        spherical_sampling: bool = False,
        reference_translation_noise_m: float = 0.0,
        reference_rotation_noise_deg: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # When True, in dual-CFG mode we not only zero satellite
        # features for the image-only / unconditioned paths but also
        # rely on cond_sample_mask so that satellite residuals are not
        # added for those samples inside the model.
        self.strict_dual_cfg_sat_mask = strict_dual_cfg_sat_mask
        self.use_sat_dpt_head = sat_use_dpt_head
        self.sat_dpt_head_config = (sat_dpt_head_config or {}).copy()
        self.train_sat_dpt_head = train_sat_dpt_head
        default_sat_dim = get_satellite_feature_dim(sat_encoder_name)
        self.sat_feature_dim = int(
            self.sat_dpt_head_config.get("n_output_channels", default_sat_dim)
        )

        if self.use_sat_dpt_head:
            raise RuntimeError("This release does not include a DPT satellite head.")
        print("Initializing satellite encoder...")
        self.sat_encoder = sat_utils.get_sat_encoder(
            sat_encoder_name,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
        )
        self.sat_encoder.to(self.device)
        if not train_sat_encoder:
            self.sat_encoder.eval()
        self.sat_encoder.requires_grad_(train_sat_encoder)
        self.sat_transform = sat_utils.get_sat_encoder_transform(sat_encoder_name)

        self.disable_sat_conditioning = disable_sat_conditioning
        self.train_sat_encoder = train_sat_encoder
        self._sample_points_min_depth_px = sample_points_min_depth
        self._sample_points_max_depth_px = sample_points_max_depth
        self._num_sample_points = num_sample_points
        if add_positional_encoding:
            self.sat_pos_encoding = PositionalEncoding(
                d_model=self.sat_feature_dim,
                max_len=64,
                scale=position_encoding_scale,
            )
        else:
            self.sat_pos_encoding = None

        # Match UNet latent resolution to input size and VAE downsampling
        self.latent_h = self.img_h // self.vae_downsampling_factor
        self.latent_w = self.img_w // self.vae_downsampling_factor

        self.ray_position_embedder = (
            sat_utils.RayPositionalEncoder(
                d_input=3,
                n_freqs=ray_pos_enc_freqs,
                log_space=True,
                include_input=False,
            )
            if ray_pos_enc_freqs > 0
            else None
        )
        self.spherical_sampling = spherical_sampling
        self.reference_translation_noise_m = float(reference_translation_noise_m)
        self.reference_rotation_noise_deg = float(reference_rotation_noise_deg)

    def _get_or_create_reference_frame_noise(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one reference-frame offset per item and cache it in-batch."""
        B = batch["images"].shape[0]
        translation_key = "_reference_frame_translation_noise_px"
        yaw_key = "_reference_frame_yaw_noise_rad"
        if translation_key in batch and yaw_key in batch:
            return (
                batch[translation_key].to(device=device, dtype=dtype),
                batch[yaw_key].to(device=device, dtype=dtype),
            )

        translation_noise_px = torch.zeros((B, 2), device=device, dtype=dtype)
        yaw_noise_rad = torch.zeros((B,), device=device, dtype=dtype)

        if self.reference_translation_noise_m > 0:
            if "satellite_ppm" not in batch:
                raise ValueError(
                    "Missing 'satellite_ppm' in batch while reference_translation_noise_m > 0. "
                    "Provide satellite ppm from dataset satellite configs."
                )
            pixels_per_meter = batch["satellite_ppm"].to(device=device, dtype=dtype).reshape(B, 1)
            direction = torch.rand((B,), device=device, dtype=dtype) * (2.0 * math.pi)
            magnitude_px = (self.reference_translation_noise_m * pixels_per_meter).squeeze(-1)
            translation_noise_px[:, 0] = magnitude_px * torch.cos(direction)
            translation_noise_px[:, 1] = magnitude_px * torch.sin(direction)

        if self.reference_rotation_noise_deg > 0:
            direction_sign = torch.where(
                torch.rand((B,), device=device) < 0.5,
                torch.full((B,), -1.0, device=device, dtype=dtype),
                torch.full((B,), 1.0, device=device, dtype=dtype),
            )
            yaw_noise_rad = (
                direction_sign
                * self.reference_rotation_noise_deg
                * math.pi
                / 180.0
            )

        batch[translation_key] = translation_noise_px
        batch[yaw_key] = yaw_noise_rad
        return translation_noise_px, yaw_noise_rad

    def encode_satellite_images(self, satellite_images):
        wrapped = False
        if satellite_images.dim() == 5:
            B, M, C, H, W = satellite_images.shape
            assert M == 1, "M > 1 not supported"
            satellite_images = satellite_images.reshape(B * M, C, H, W)
            wrapped = True
        satellite_images = satellite_images.to(self.device)
        satellite_images = self.sat_transform(satellite_images)
        satellite_tokens = self.sat_encoder.forward_features(satellite_images)["x_norm_patchtokens"]
        sat_emb_size = int(math.sqrt(satellite_tokens.shape[-2]))
        if wrapped:
            satellite_tokens = satellite_tokens.reshape(
                B, -1, sat_emb_size, sat_emb_size, satellite_tokens.shape[-1]
            )
        else:
            satellite_tokens = satellite_tokens.reshape(
                -1, sat_emb_size, sat_emb_size, satellite_tokens.shape[-1]
            )
        return satellite_tokens

    def _sample_ray_points(self, batch, latent_ds_factor=8):
        # Pick a row of the image, and sample points along a ray for each pixel.
        B, N = batch["images"].shape[:2]
        width = batch["images"].shape[-1] // latent_ds_factor
        u = torch.arange(width, device=self.device)[None, None] + 0.5
        intrinsics = batch["intrinsics"].clone()
        intrinsics = intrinsics.to(self.device)
        intrinsics_scale = 1.0 / latent_ds_factor
        intrinsics[..., :2, :3] *= intrinsics_scale
        x = (u - intrinsics[:, :, 0, 2, None]) / intrinsics[:, :, 0, 0, None]  # B, N, latents_W
        x = x.unsqueeze(-1).repeat(1, 1, 1, self._num_sample_points)  # B, N, latents_W, num_points
        z_vals = torch.linspace(
            self._sample_points_min_depth_px,
            self._sample_points_max_depth_px,
            self._num_sample_points,
            device=self.device,
        )
        z_vals = z_vals[None, None, None, :].repeat(B, N, width, 1)  # B, N, latents_W, num_points
        ray_points = torch.stack([x, -torch.ones_like(x)], dim=-1)  # B, N, latents_W, num_points, 2
        if self.spherical_sampling:
            ray_points = ray_points / torch.norm(ray_points, dim=-1, keepdim=True)
        ray_points = ray_points * z_vals[..., None]
        return ray_points

    def _get_transformed_ray_points(self, batch, latent_ds_factor=8):
        # Get the features for the ray points.
        ray_points = self._sample_ray_points(batch, latent_ds_factor=latent_ds_factor)
        cam_poses = batch["camera_poses"].to(ray_points.device)
        ref_rot = batch["reference_rotation"].to(ray_points.device)
        ref_translation_noise_px, ref_yaw_noise_rad = self._get_or_create_reference_frame_noise(
            batch=batch, device=ray_points.device, dtype=ray_points.dtype
        )

        # Apply yaw perturbation in the world frame. This keeps camera-to-camera
        # relative poses unchanged while avoiding dataset-specific assumptions
        # about reference-frame axis conventions.
        if torch.any(ref_yaw_noise_rad != 0):
            c = torch.cos(ref_yaw_noise_rad)
            s = torch.sin(ref_yaw_noise_rad)
            yaw_offset_rot = torch.zeros(
                (ref_rot.shape[0], 3, 3), device=ref_rot.device, dtype=ref_rot.dtype
            )
            yaw_offset_rot[:, 0, 0] = c
            yaw_offset_rot[:, 0, 2] = s
            yaw_offset_rot[:, 1, 1] = 1.0
            yaw_offset_rot[:, 2, 0] = -s
            yaw_offset_rot[:, 2, 2] = c
            ref_rot = yaw_offset_rot @ ref_rot

        view_ref_R_cam = ref_rot[:, None].transpose(-1, -2) @ cam_poses[..., :3, :3]  # B, N, 3, 3
        z_rot_clock = torch.atan2(view_ref_R_cam[:, :, 0, 2], view_ref_R_cam[:, :, 2, 2])
        img_center_rot_clock = z_rot_clock
        c, s = torch.cos(img_center_rot_clock), torch.sin(img_center_rot_clock)
        rot_mat = torch.stack(
            [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-1
        )  # B, N, 2, 2

        ray_points = (
            rot_mat[:, :, None, None] @ ray_points[..., None]
        )  # B, N, latents_W, num_points, 2, 1
        ray_points = ray_points.squeeze(-1)  # B, N, latents_W, num_points, 2
        sat_res = batch["satellite_images"].shape[-1]
        ray_points = ray_points + torch.tensor(
            [sat_res / 2.0, sat_res / 2.0], device=ray_points.device
        )
        # This is hacky, only works for 4 ppm, also the translation reference not the same as the rotation reference
        if "reference_translation" in batch:
            center_translation = batch["reference_translation"][:, None, :2].to(
                ray_points.device
            )  # B, N, 2
            translation_offset = cam_poses[..., :2, 3] - center_translation  # B, N, 2
            # convert to pixels
            translation_offset = translation_offset * 4
            ray_points = (
                ray_points
                + torch.stack([translation_offset[..., 0], -translation_offset[..., 1]], dim=-1)[
                    :, :, None, None, :
                ]
            )  # B, N, lat
        # Add one shared translation perturbation in satellite-image coordinates.
        ray_points = ray_points + ref_translation_noise_px[:, None, None, None, :]
        return ray_points

    def _get_ray_features(self, batch, satellite_features: torch.Tensor, latent_ds_factor=8):
        # Get the features for the ray points.
        ray_points = self._get_transformed_ray_points(batch, latent_ds_factor=latent_ds_factor)

        B, N = batch["images"].shape[:2]
        satellite_features = satellite_features.expand(-1, N, -1, -1, -1)
        satellite_features = rearrange(satellite_features, "b n h w c -> (b n) c h w")
        # Ray points are in the range [-1, 1]
        ray_points = rearrange(ray_points, "b n w z d -> (b n) w z d")
        sat_res = batch["satellite_images"].shape[-1]
        ray_points = 2 * ray_points / sat_res - 1
        ray_features = F.grid_sample(satellite_features, ray_points, align_corners=True)
        ray_features = rearrange(ray_features, "bn c w z -> bn w z c")
        return ray_features, ray_points

    def update_inputs_ldm_kwargs_for_timestep(
        self,
        scaled_conditioned_input,
        ldm_kwargs,
        sigma_hat,
        N,
        cross_attn_condn,
        dense_condn,
        uncond_input_mask=None,
        is_cfg=False,
    ):
        return scaled_conditioned_input, ldm_kwargs, N, cross_attn_condn, dense_condn

    def additional_ldm_conditions(
        self,
        batch,
        input_latents: torch.Tensor,
        uncond_input_mask: torch.Tensor | None = None,
        is_cfg: bool = False,
        debug: bool = False,
        num_noise_samples: int = 1,
    ):
        batch["satellite_images"] = batch["satellite_images"].to(self.device)
        scope = nullcontext() if self.train_sat_encoder else torch.no_grad()
        with scope:
            sat_cond = self.encode_satellite_images(batch["satellite_images"])

        B, N = batch["images"].shape[:2]

        sat_ray_features_dict = {}
        sat_ray_points_dict = {}
        street_raymap_dict = {}
        street_raymap_raw_dict = {}
        cond_sample_mask = None

        num_sat = sat_cond.shape[1]
        for i in range(4):
            sat_ray_features, sat_ray_points = self._get_ray_features(
                batch, sat_cond, latent_ds_factor=(2**i) * self.vae_downsampling_factor
            )
            w = sat_ray_features.shape[1]
            sat_ray_features = rearrange(sat_ray_features, "bn w z c -> (bn w) z c")
            if self.sat_pos_encoding is not None:
                sat_ray_features = self.sat_pos_encoding(sat_ray_features)
            sat_ray_features = rearrange(sat_ray_features, "(b n w) z c -> b n w z c", w=w, n=N)

            if uncond_input_mask is not None:
                sat_ray_features = torch.where(
                    uncond_input_mask[:, None, None, None, None],
                    torch.zeros_like(sat_ray_features),
                    sat_ray_features,
                )
            sat_ray_features = rearrange(sat_ray_features, "b n w z c -> (b n) w z c")

            if self.disable_sat_conditioning:
                sat_ray_features = torch.zeros_like(sat_ray_features)

            ray_directions = sat_utils.compute_yaw_aligned_ray_directions(
                batch,
                latent_H=self.latent_h // (2**i),
                latent_W=self.latent_w // (2**i),
                vae_downsampling_factor=(2**i) * self.vae_downsampling_factor,
                device=self.device,
            )
            ray_directions = rearrange(ray_directions, "b n h w d -> (b n) (h w) d")
            # Keep a pre-featureized 3D raymap copy for debugging/comparison.
            ray_directions_raw = ray_directions
            if self.ray_position_embedder is not None:
                ray_directions = self.ray_position_embedder(ray_directions)

            if num_noise_samples > 1:
                sat_ray_features = sat_ray_features[None].repeat(
                    num_noise_samples, *(1,) * sat_ray_features.ndim
                )
                sat_ray_features = sat_ray_features.reshape(-1, *sat_ray_features.shape[2:])
                ray_directions = ray_directions[None].repeat(
                    num_noise_samples, *(1,) * ray_directions.ndim
                )
                ray_directions = ray_directions.reshape(-1, *ray_directions.shape[2:])
                ray_directions_raw = ray_directions_raw[None].repeat(
                    num_noise_samples, *(1,) * ray_directions_raw.ndim
                )
                ray_directions_raw = ray_directions_raw.reshape(-1, *ray_directions_raw.shape[2:])
                sat_ray_points = sat_ray_points[None].repeat(
                    num_noise_samples, *(1,) * sat_ray_points.ndim
                )
                sat_ray_points = sat_ray_points.reshape(-1, *sat_ray_points.shape[2:])

            # 1 for conditional samples, 0 for unconditional samples in CFG batch.
            level_cond_sample_mask = torch.ones(
                (sat_ray_features.shape[0],),
                device=sat_ray_features.device,
                dtype=sat_ray_features.dtype,
            )
            # --- Conditional batching logic ---
            if is_cfg:
                if self.use_dual_cfg:
                    # 3-way batch: [sat_features, zeros, zeros]
                    sat_ray_features = torch.cat(
                        [
                            sat_ray_features,
                            torch.zeros_like(sat_ray_features),
                            torch.zeros_like(sat_ray_features),
                        ],
                        dim=0,
                    )
                    level_cond_sample_mask = torch.cat(
                        [
                            level_cond_sample_mask,
                            torch.zeros_like(level_cond_sample_mask),
                            torch.zeros_like(level_cond_sample_mask),
                        ],
                        dim=0,
                    )
                    ray_directions = torch.cat([ray_directions] * 3, dim=0)
                    ray_directions_raw = torch.cat([ray_directions_raw] * 3, dim=0)
                    sat_ray_points = torch.cat([sat_ray_points] * 3, dim=0)
                else:
                    # 2-way batch: [sat_features, zeros]
                    sat_ray_features = torch.cat(
                        [sat_ray_features, torch.zeros_like(sat_ray_features)], dim=0
                    )
                    level_cond_sample_mask = torch.cat(
                        [level_cond_sample_mask, torch.zeros_like(level_cond_sample_mask)], dim=0
                    )
                    ray_directions = torch.cat([ray_directions, ray_directions], dim=0)
                    ray_directions_raw = torch.cat([ray_directions_raw, ray_directions_raw], dim=0)
                    sat_ray_points = torch.cat([sat_ray_points, sat_ray_points], dim=0)
            # ------------------------------------
            cond_sample_mask = level_cond_sample_mask

            sat_ray_features_dict[2**i] = sat_ray_features
            sat_ray_points_dict[2**i] = sat_ray_points

            street_raymap_dict[2**i] = ray_directions
            street_raymap_raw_dict[2**i] = ray_directions_raw

        if debug:
            return {
                "sat_cond": sat_ray_features_dict,
                "debug_sat_cond": sat_cond,
                "debug_ray_points": sat_ray_points_dict,
                "street_raymap_for_res": street_raymap_dict,
                "street_raymap_raw_for_res": street_raymap_raw_dict,
            }
        # Always zero out satellite features for the non-satellite
        # conditioned paths in CFG batches. Optionally, when
        # strict_dual_cfg_sat_mask is enabled, also pass through
        # cond_sample_mask so that satellite residuals are completely
        # suppressed for those paths inside the satellite model.
        ldm_kwargs = {
            "sat_cond": sat_ray_features_dict,
            "street_raymap_for_res": street_raymap_dict,
            "street_raymap_raw_for_res": street_raymap_raw_dict,
        }
        if self.strict_dual_cfg_sat_mask and cond_sample_mask is not None:
            ldm_kwargs["cond_sample_mask"] = cond_sample_mask
        return ldm_kwargs

    def debug_images(self, batch, sat_attn_vals=None):

        base_sat_img = batch["satellite_images"][0, 0]
        num_views = batch["images"].shape[1]

        def _build_images(attention_map=None):
            latent_ds_factor = 8
            if attention_map is not None:
                resolution_W = attention_map.shape[-2]  # assume B, N, W, Z
                latent_ds_factor = latent_ds_factor * (self.latent_w // resolution_W)

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                ray_points = self._get_transformed_ray_points(
                    batch, latent_ds_factor=latent_ds_factor
                )

            imgs = []
            for v in range(num_views):
                attn_slice = None
                if attention_map is not None and v < attention_map.shape[1]:
                    attn_slice = attention_map[0, v]
                imgs.append(
                    viz_utils.visualize_ray_points(
                        base_sat_img,
                        ray_points[0, v],
                        attention_values=attn_slice,
                    )
                )
            imgs = np.stack(imgs, axis=0)
            return torch.from_numpy(imgs).permute(0, 3, 1, 2).to(self.device) / 255.0 * 2 - 1

        if sat_attn_vals:
            colored_outputs = {}
            for key, attn in sat_attn_vals.items():
                colored_outputs[key] = _build_images(attn)
            return colored_outputs

        return _build_images()

    def debug_fov_triangles(self, batch, fov_depth=50):
        """
        Visualizes camera FOV triangles on satellite images.
        Creates one image per target view, showing all input views in one color
        and the target view in another color.

        Args:
            batch: Batch dictionary with camera poses, intrinsics, input_mask, target_mask
            fov_depth: Depth at which to compute the FOV triangle edges (in pixels)

        Returns:
            torch.Tensor: Images with FOV triangles drawn, shape (num_target_views, C, H, W)
        """
        return viz_utils.create_fov_triangle_visualizations(
            batch,
            device=self.device,
            fov_depth=fov_depth,
            camera_pos_depth=self._sample_points_min_depth_px,
        )
