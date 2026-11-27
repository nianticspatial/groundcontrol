from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import groundcontrol.training.geometry as geometry_utils


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class SevaPipeline(nn.Module):
    """
    Pipeline for SEVA-style multi-view inference, corrected to match the specific
    model API and input preparation logic.
    """

    def __init__(
        self,
        ldm,  # The main UNet-like model
        vae,
        semantic_encoder,
        scheduler,
        default_denoising_steps: int = 1000,
        vae_downsampling_factor: int = 8,
        cfg_scale: float = 1.0,  # Not fully implemented, placeholder
        s_churn: float = 0.0,
        device="cpu",
        dtype=None,
        using_older_ckpt: bool = True,
        use_reference_rotation: bool = False,
        return_attn: bool = False,
        drop_clip_conditioning: bool = False,
        img_h: int = 256,
        img_w: int = 256,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        super().__init__()

        self.ldm = ldm
        self.vae = vae
        self.scheduler = scheduler
        self.semantic_encoder = semantic_encoder
        # TODO: Use the standard diffusers `register_modules` to handle device and dtype
        # self.register_modules(
        #     ldm=ldm,
        #     vae=vae,
        #     scheduler=scheduler, # `scheduler` is the standard name
        #     semantic_encoder=semantic_encoder,
        # )

        # Store additional config that is not a torch.nn.Module
        self.vae_downsampling_factor = vae_downsampling_factor
        self.cfg_scale = cfg_scale
        self.s_churn = s_churn
        self.default_denoising_steps = default_denoising_steps
        self.device = device
        if dtype is not None:
            self.ldm.to(dtype)
            self.vae.to(dtype)
            self.scheduler.train_sigmas.to(dtype)
            if self.semantic_encoder is not None:
                self.semantic_encoder.to(dtype)

        self.ldm.to(device)
        self.vae.to(device)
        if self.semantic_encoder is not None:
            self.semantic_encoder.to(device)
        self.scheduler.train_sigmas.to(device)
        self.dtype = dtype

        self.use_reference_rotation = use_reference_rotation
        self.using_older_ckpt = using_older_ckpt
        self.return_attn = return_attn
        self.drop_clip_conditioning = drop_clip_conditioning
        self.img_h = img_h
        self.img_w = img_w
        # Store generator for reproducible noise generation (None = use default random state)
        self.generator = generator

    @staticmethod
    def _tensor_to_debug_numpy(tensor: torch.Tensor) -> np.ndarray:
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()

    def _maybe_dump_plucker_trace(
        self, stage: str, arrays: dict[str, torch.Tensor | np.ndarray | int]
    ) -> None:
        if not getattr(self, "debug_dump_plucker_trace", False):
            return
        dump_dir = getattr(self, "debug_dump_plucker_trace_dir", None)
        if not dump_dir:
            return
        max_dumps = int(getattr(self, "debug_dump_plucker_trace_max_dumps", 100))
        dump_count = int(getattr(self, "_debug_dump_plucker_trace_count", 0))
        if dump_count >= max_dumps:
            return

        sample_tag = str(getattr(self, "_debug_current_sample_id", "unknown")).replace("/", "_")
        sample_tag = sample_tag.replace(" ", "_")
        sample_tag = sample_tag[:120]

        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dump_count:04d}_{stage}_{sample_tag}.npz"

        to_save: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            if isinstance(value, torch.Tensor):
                to_save[key] = self._tensor_to_debug_numpy(value)
            elif isinstance(value, np.ndarray):
                to_save[key] = value
            else:
                to_save[key] = np.asarray([value])
        np.savez_compressed(output_path, **to_save)
        setattr(self, "_debug_dump_plucker_trace_count", dump_count + 1)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes images to the latent space using the VAE."""
        is_5d = images.ndim == 5
        if is_5d:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)

        # Note: Using self.vae as registered in __init__
        latents = self.vae.encode(images)

        if is_5d:
            latents = latents.view(B, N, *latents.shape[1:])
        return latents

    @torch.no_grad()
    def prepare_conditioning(
        self,
        images,
        input_mask,
        poses,
        intrinsics,
        inlcude_latents: bool = True,
        reference_rotation: torch.Tensor | None = None,
    ) -> dict:
        """
        Creates all conditioning tensors for the model in a vectorized manner,
        adhering to the user's specified logic.
        """
        B, N = poses.shape[:2]
        # Enforce consistency with configured image size
        H = self.img_h
        W = self.img_w
        assert (
            images.shape[-2] == H and images.shape[-1] == W
        ), f"Input image size {(images.shape[-2], images.shape[-1])} does not match config (img_h, img_w)=({H}, {W})"
        latent_size = (
            H // self.vae_downsampling_factor,
            W // self.vae_downsampling_factor,
        )
        # Normalize x by image width and y by image height
        intrinsics_normalizer = torch.tensor(
            [W, H, 1.0], device=intrinsics.device, dtype=intrinsics.dtype
        )
        normalized_intrinsics = intrinsics / intrinsics_normalizer[None, :, None]

        poses_flat = poses.view(B * N, 4, 4)
        intrinsics_flat = normalized_intrinsics.view(B * N, 3, 3)

        if self.using_older_ckpt:
            c2w_src = poses[:, 0, :, :].clone()
        else:
            c2w_src = torch.eye(4).to(poses).unsqueeze(0).repeat(B, 1, 1)
            c2w_src[:, :3, :3] = (
                poses[:, 0, :3, :3] if reference_rotation is None else reference_rotation
            )
            # c2w_src[:, :3, 3] = poses[:, 0, :3, 3]

        c2w_src_expanded = c2w_src.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 4, 4)

        # Optional deep trace dump around Plucker construction.
        if getattr(self, "debug_dump_plucker_trace", False):
            w2c_all = torch.linalg.inv(poses_flat.float()).to(poses_flat.dtype)
            extrinsics_rel = torch.einsum("vnm,vmp->vnp", w2c_all, c2w_src_expanded)
            self._maybe_dump_plucker_trace(
                "plucker_inputs",
                {
                    "poses": poses,
                    "poses_flat": poses_flat,
                    "intrinsics": intrinsics,
                    "normalized_intrinsics": normalized_intrinsics,
                    "intrinsics_flat": intrinsics_flat,
                    "c2w_src": c2w_src,
                    "c2w_src_expanded": c2w_src_expanded,
                    "extrinsics_rel": extrinsics_rel,
                    "input_mask": input_mask,
                    "B": B,
                    "N": N,
                },
            )

        plucker_coords = geometry_utils.get_plucker_coordinates(
            extrinsics_src=c2w_src_expanded,
            extrinsics=poses_flat,
            intrinsics=intrinsics_flat,
            target_size=latent_size,
            is_extrinsics_c2w=True,
            input_is_batched=True,
        )
        plucker_coords = plucker_coords.view(B, N, *plucker_coords.shape[1:]).to(images)

        if getattr(self, "debug_dump_plucker_trace", False):
            moments = plucker_coords[:, :, 3:6]
            moment_norms = torch.linalg.norm(moments, dim=2)  # [B, N, H, W]
            view_moment_mean = moment_norms.mean(dim=(-1, -2))
            view_moment_max = moment_norms.amax(dim=(-1, -2))
            self._maybe_dump_plucker_trace(
                "plucker_outputs",
                {
                    "plucker_coords": plucker_coords,
                    "moment_norms": moment_norms,
                    "view_moment_mean": view_moment_mean,
                    "view_moment_max": view_moment_max,
                },
            )

        input_mask_image = (
            input_mask[..., None, None, None].expand(-1, -1, -1, *plucker_coords.shape[-2:]).clone()
        )
        concat_condition = torch.cat([input_mask_image.float(), plucker_coords], dim=2)

        if (
            not self.drop_clip_conditioning
            and self.semantic_encoder is not None
            and input_mask.any().item()
        ):
            self.semantic_encoder.to(torch.bfloat16)
            images_flat = images[input_mask].to(torch.bfloat16)
            image_embeds = self.semantic_encoder(images_flat)
            num_inputs_per_item = input_mask.sum(dim=1).tolist()
            image_embeds_list = torch.split(image_embeds, num_inputs_per_item)
            mean_embeds = [
                torch.sum(embed, dim=0) / (num_inputs_per_item[i] + 1e-8)
                for i, embed in enumerate(image_embeds_list)
            ]
            mean_embeds = torch.stack(mean_embeds, dim=0)
            img_cond_mask = input_mask.sum(dim=1) > 0
            cross_attn_cond = torch.zeros((B, mean_embeds.shape[-1]), device=mean_embeds.device).to(
                mean_embeds
            )
            cross_attn_cond[img_cond_mask] = mean_embeds[img_cond_mask]
            cross_attn_cond = einops.repeat(cross_attn_cond, "b d -> b n 1 d", n=N)
        else:
            cross_attn_cond = (
                torch.zeros((1, 1, 1, self.ldm.params.context_dim), device=self.device)
                .to(torch.bfloat16)
                .expand(B, N, 1, -1)
                .clone()
            )

        if inlcude_latents and input_mask.any():
            input_latents = self.encode_images(images[input_mask])
        else:
            input_latents = None

        conditioning = {
            "cross_attn_cond": cross_attn_cond,
            "plucker_coords": plucker_coords,
            "concat_condition": concat_condition,
            "input_mask_image": input_mask_image,
        }
        if inlcude_latents:
            conditioning["input_latents"] = input_latents

        return conditioning

    def init_unconditionional(self, cross_attn_cond, concat_condition, plucker_coords):
        uncond_cross_attn_cond = torch.zeros_like(cross_attn_cond)
        uncond_concat_condition = torch.cat(
            [torch.zeros_like(concat_condition[:, :, :1]), concat_condition[:, :, 1:]],
            dim=2,
        )
        uncond_plucker_coords = plucker_coords
        return uncond_cross_attn_cond, uncond_concat_condition, uncond_plucker_coords

    def prepare_inputs_cfg(
        self, model_input, noisy_latents, concat_condn, plucker_coords, cross_attn_condn
    ):
        # semantic conditioning, unconditioned is jus zero.
        cross_attn_condn_input = torch.cat(
            [cross_attn_condn, torch.zeros_like(cross_attn_condn)], dim=0
        )

        # "concatenated condition": input_mask + camera poses
        # we treat the input mask for unconditioned sample as all zeros.
        concat_condn_input = torch.cat(
            [
                concat_condn,
                torch.cat(
                    [torch.zeros_like(concat_condn[:, :, :1]), concat_condn[:, :, 1:]],
                    dim=2,
                ),
            ],
            dim=0,
        )

        dense_vector_condn = torch.cat([plucker_coords] * 2, dim=0)

        # model input may contain pure latents for known images, which is the replace condition
        model_input = torch.cat([model_input, noisy_latents], dim=0)
        return (
            model_input,
            cross_attn_condn_input,
            concat_condn_input,
            dense_vector_condn,
        )

    def get_output_cfg(self, predicted_output, cfg_scale):
        conditioned_output, unconditioned_output = predicted_output.chunk(2)
        return unconditioned_output + cfg_scale * (conditioned_output - unconditioned_output)

    def normalize_camera_poses_training(
        self, poses, valid_mask, default_scale=2.0, divison_factor=None
    ):
        # Create a copy to avoid inplace operations on inference tensors
        poses_input = poses
        poses = poses.clone()
        # Compute the per-sample centroid
        # TODO: assuming right now the first camera is valid, pass in a new parameter if not later.
        mask = valid_mask[..., None].float()
        num_valid_views = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        centroid = (poses[..., :3, 3] * mask).sum(dim=1, keepdim=True) / num_valid_views
        poses[..., :3, 3] -= centroid

        distances = torch.norm(poses[..., :3, 3], dim=-1, keepdim=True)
        max_distances = torch.max(distances, dim=1, keepdim=True).values
        # TODO: the loss becomes nan when using max_distances = distances[:, :1].clone(), investigate why.
        # Seva does this during inference.
        # max_distances = distances[:, :1].clone()

        normalization = max_distances if divison_factor is None else max_distances / divison_factor
        translation_scaling_factor = torch.where(
            torch.isclose(max_distances, torch.zeros_like(max_distances), atol=1e-5),
            default_scale,
            default_scale / normalization,
        )

        poses[..., :3, 3] *= translation_scaling_factor

        if getattr(self, "debug_dump_plucker_trace", False):
            self._maybe_dump_plucker_trace(
                "pose_normalization",
                {
                    "poses_input": poses_input,
                    "poses_normalized": poses,
                    "valid_mask": valid_mask,
                    "translation_scaling_factor": translation_scaling_factor,
                    "default_scale": default_scale,
                },
            )

        return poses, translation_scaling_factor

    def reshape_model_inputs(self, concat_model_input, dense_condn, cross_attn_condn, timestep_val):
        N = concat_model_input.shape[1]  # number of frames
        final_input_flat = concat_model_input.view(-1, *concat_model_input.shape[2:])
        cross_attn_condn = cross_attn_condn.reshape(-1, 1, cross_attn_condn.shape[-1])
        dense_condn = dense_condn.view(-1, *dense_condn.shape[2:])
        timestep_val = timestep_val.repeat_interleave(N)
        cross_attn_condn = cross_attn_condn.to(final_input_flat)
        dense_condn = dense_condn.to(final_input_flat)
        return final_input_flat, cross_attn_condn, dense_condn, timestep_val

    def scale_and_camera_condition_input(self, model_input, sigma, concat_condn):
        model_input_scaled = model_input * (
            1 / (append_dims(sigma, model_input.ndim) ** 2 + 1.0) ** 0.5
        )
        return torch.cat([model_input_scaled, concat_condn], dim=2)

    def convert_model_output_to_x0(self, model_output, sigmas_t, noisy_latents):
        # model output is epsilon_pred
        return noisy_latents - sigmas_t * model_output

    def decode_latents(self, latents):
        B, N, C, H, W = latents.shape
        result = self.vae.decode(latents.view(-1, *latents.shape[2:]))
        return result.view(B, N, *result.shape[1:])

    def infer_target_images(
        self,
        input_images,
        input_poses,
        target_poses,
        input_intrinsics,
        target_intrinsics,
        num_inference_steps=None,
        num_noises_per_frame=1,
    ):
        target_images = torch.zeros(target_poses.shape[0], *input_images.shape[1:])
        all_images = torch.cat([input_images, target_images], dim=0)
        all_poses = torch.cat([input_poses, target_poses], dim=0)
        all_intrinsics = torch.cat([input_intrinsics, target_intrinsics], dim=0)
        input_mask = torch.cat(
            [
                torch.ones(input_images.shape[0], dtype=torch.bool),
                torch.zeros(target_images.shape[0], dtype=torch.bool),
            ],
            dim=0,
        )
        target_mask = torch.cat(
            [
                torch.zeros(input_images.shape[0], dtype=torch.bool),
                torch.ones(target_images.shape[0], dtype=torch.bool),
            ],
            dim=0,
        )
        batch = {
            "images": all_images,
            "input_mask": input_mask,
            "target_mask": target_mask,
            "camera_poses": all_poses,
            "intrinsics": all_intrinsics,
        }
        return self(
            batch,
            num_inference_steps=num_inference_steps,
            num_noises_per_frame=num_noises_per_frame,
        )

    def additional_ldm_conditions(
        self,
        batch,
        input_latents: torch.Tensor,
        uncond_input_mask: torch.Tensor | None = None,
        is_cfg: bool = False,
        num_noise_samples: int = 1,
    ):
        return {}

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

    def update_epsilon_pred_for_timestep(self, epsilon_pred, num_frames):
        return epsilon_pred, 0.0

    @torch.no_grad()
    def run_denoising_loop(
        self,
        input_mask,
        target_mask,
        normalized_poses,
        conditioning,
        num_inference_steps=None,
        num_noises_per_frame=1,
        return_x0_list=False,
        init_noise=None,
        **ldm_kwargs,
    ):
        B, N = input_mask.shape[:2]
        latent_H = self.img_h // self.vae_downsampling_factor
        latent_W = self.img_w // self.vae_downsampling_factor
        if num_noises_per_frame > 1:
            # add duplicate batches to all inputs
            input_mask = torch.cat([input_mask] * num_noises_per_frame, dim=0).clone()
            target_mask = torch.cat([target_mask] * num_noises_per_frame, dim=0).clone()
            normalized_poses = torch.cat([normalized_poses] * num_noises_per_frame, dim=0).clone()
            for k, val in conditioning.items():
                if val is None:
                    continue
                conditioning[k] = val.repeat(num_noises_per_frame, *(1,) * (val.ndim - 1))

        x_0_list = []
        x_0_uncond_list = []
        x_0_cond_list = []

        with torch.inference_mode():
            latents_shape = (
                B * num_noises_per_frame,
                N,
                self.ldm.out_channels,
                latent_H,
                latent_W,
            )
            if init_noise is None:
                # Use stored generator for reproducible noise generation if provided
                if self.generator is not None:
                    init_noise = torch.randn(
                        latents_shape, generator=self.generator, device=self.device
                    )
                else:
                    init_noise = torch.randn(latents_shape, device=self.device)

            latents = init_noise.clone()
            if self.dtype is not None:
                latents = latents.to(self.dtype)

            # 3. Set timesteps and scale initial noise
            self.scheduler.set_timesteps(
                num_inference_steps,
                dtype=self.dtype if self.dtype is not None else torch.float32,
            )
            latents *= self.scheduler.init_noise_sigma
            input_latent_noise = None
            if input_mask.any():
                if self.generator is not None:
                    input_latent_noise = torch.randn(
                        conditioning["input_latents"].shape,
                        generator=self.generator,
                        device=conditioning["input_latents"].device,
                        dtype=conditioning["input_latents"].dtype,
                    )
                else:
                    input_latent_noise = torch.randn_like(conditioning["input_latents"])

            # Only create debug_dicts list if we actually need it
            all_debug_dicts = [] if self.return_attn else None

            # 4. The Denoising Loop
            for i, t in tqdm(
                enumerate(self.scheduler.timesteps),
                total=num_inference_steps,
                desc="Denoising...",
            ):
                sigma = self.scheduler.sampling_sigmas[i]

                # 4a. Apply stochastic churn if needed
                gamma = (
                    min(self.s_churn / num_inference_steps, 2**0.5 - 1)
                    if 0 < sigma < float("inf")
                    else 0.0
                )
                sigma_hat = sigma * (gamma + 1.0)

                churned_latents = latents
                if gamma > 0:
                    if self.generator is not None:
                        eps = (
                            torch.randn(
                                latents.shape,
                                generator=self.generator,
                                device=latents.device,
                                dtype=latents.dtype,
                            )
                            * 1.0
                        )  # s_noise
                    else:
                        eps = torch.randn_like(latents) * 1.0  # s_noise
                    churning = eps * torch.sqrt(sigma_hat**2 - sigma**2)
                    churned_latents += churning
                else:
                    churning = torch.zeros_like(latents)

                train_t = self.scheduler.sigma_to_training_idx(sigma_hat)
                sigma_hat = self.scheduler.training_idx_to_sigma(train_t)

                # 4b. Prepare model input by replacing known views and concatenating conditions
                noisy_latents = churned_latents.clone()

                # Use torch.where instead of inplace assignment to avoid inference tensor issues
                if input_mask.any():
                    noisy_latents[input_mask] = conditioning["input_latents"].to(noisy_latents)
                    churned_latents[input_mask] = (
                        self.scheduler.add_noise(
                            conditioning["input_latents"],
                            input_latent_noise,
                            train_t.expand(
                                input_latent_noise.shape[0],
                            ),
                        )
                        + churning[input_mask]
                    )

                noisy_latents, cross_attn_condn, concat_condn, dense_condn = (
                    self.prepare_inputs_cfg(
                        noisy_latents,
                        churned_latents,
                        conditioning["concat_condition"],
                        conditioning["plucker_coords"],
                        conditioning["cross_attn_cond"],
                    )
                )

                scaled_conditioned_input = self.scale_and_camera_condition_input(
                    noisy_latents, sigma_hat, concat_condn
                )
                train_t = (
                    torch.full((scaled_conditioned_input.shape[0],), train_t)
                    .to(scaled_conditioned_input)
                    .int()
                )

                with torch.autocast(device_type="cuda"):
                    (
                        scaled_conditioned_input,
                        updated_ldm_kwargs,
                        num_frames,
                        cross_attn_condn,
                        dense_condn,
                    ) = self.update_inputs_ldm_kwargs_for_timestep(
                        scaled_conditioned_input,
                        ldm_kwargs,
                        sigma_hat,
                        N,
                        cross_attn_condn,
                        dense_condn,
                        is_cfg=True,
                    )
                    (
                        scaled_conditioned_input,
                        cross_attn_condn,
                        dense_condn,
                        train_t,
                    ) = self.reshape_model_inputs(
                        scaled_conditioned_input,
                        dense_condn,
                        cross_attn_condn,
                        train_t,
                    )

                    debug_dict = {} if self.return_attn and i in [10, 25, 40] else None
                    epsilon_pred = self.ldm(
                        scaled_conditioned_input,
                        t=train_t,
                        y=cross_attn_condn,
                        dense_y=dense_condn,
                        num_frames=num_frames,
                        dump_step_idx=i,
                        debug_dict=debug_dict,
                        input_mask=input_mask.reshape(-1),
                        **updated_ldm_kwargs,
                    )
                    if not isinstance(epsilon_pred, torch.Tensor):
                        epsilon_pred = epsilon_pred[0]
                    epsilon_pred, _ = self.update_epsilon_pred_for_timestep(
                        epsilon_pred,
                        num_frames,
                    )
                    if debug_dict is not None and all_debug_dicts is not None:
                        all_debug_dicts.append(debug_dict)

                epsilon_pred = epsilon_pred.view(-1, N, *epsilon_pred.shape[1:])

                # 4d. Inline Euler EDM step (replaces scheduler.step)
                # Get the next sigma from the schedule
                next_sigma = self.scheduler.sampling_sigmas[i + 1]

                # Convert model output (epsilon) to predicted x_0
                # x_0 = x_t - sigma * epsilon
                # TODO: this seems like its model_input, but wondering if it should be churned_latents instead.
                pred_x0 = noisy_latents - sigma_hat.float() * epsilon_pred.float()

                # if return_x0_list:
                #     x0_images = self.vae.decode(pred_x0.view(-1, *pred_x0.shape[2:]))
                #     x0_images = x0_images.view(-1, N, *x0_images.shape[1:])
                #     cond_x0, uncond_x0 = torch.split(x0_images, 2, dim=0)
                #     x_0_uncond_list.append(uncond_x0.cpu())
                #     x_0_cond_list.append(cond_x0.cpu())
                # Clean up intermediate CPU tensors
                # del x0_images, cond_x0, uncond_x0

                # TODO: apply cfg on pred_x0
                pred_x0 = self.get_output_cfg(pred_x0, self.cfg_scale)
                # if return_x0_list:
                #     x_0_list.append(pred_x0.cpu().clone())

                with torch.autocast(device_type="cuda", enabled=False):
                    # Calculate ODE derivative: d = (x_t - x_0) / sigma_t
                    derivative = (churned_latents.float() - pred_x0) / sigma_hat.float()
                    # Calculate step size: dt = sigma_t-1 - sigma_t
                    dt = next_sigma - sigma_hat
                    # Update latents: x_t-1 = x_t + d * dt
                    new_latents = churned_latents + derivative * dt

                # Explicitly delete intermediate tensors to free memory before reassignment
                del noisy_latents, epsilon_pred, pred_x0, derivative
                if gamma > 0:
                    del eps, churning
                # Only delete churned_latents if it's not the same object as latents
                # (it might be when gamma == 0, but gets modified in-place later)
                if churned_latents is not new_latents:
                    del churned_latents
                latents = new_latents

        # 5. Decode the final latents into images
        images_flat = self.vae.decode(
            latents.view(B * num_noises_per_frame * N, *latents.shape[2:])
        )
        images = images_flat.view(B * num_noises_per_frame, N, *images_flat.shape[1:])
        images = torch.clamp(images, -1, 1)

        if num_noises_per_frame > 1:
            # remove duplicate batches
            images = images.view(num_noises_per_frame, -1, *images.shape[1:])

        # Clear intermediate variables before returning
        del latents
        # if return_x0_list:
        #     del x_0_list, x_0_uncond_list, x_0_cond_list

        # Ensure all_debug_dicts is a list (even if empty) for consistent return type
        if all_debug_dicts is None:
            all_debug_dicts = []

        return images, all_debug_dicts, init_noise

    @torch.no_grad()
    def __call__(
        self,
        batch,
        num_inference_steps=None,
        num_noises_per_frame=1,
        init_noise=None,
        pose_scale_factor=2.0,
        divison_factor=None,
    ):
        num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else self.default_denoising_steps
        )

        # 1. Prepare batch and conditioning signals
        images = batch["images"].to(self.device)
        input_mask = batch["input_mask"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)
        poses = batch["camera_poses"].to(self.device)
        intrinsics = batch["intrinsics"].to(self.device)
        reference_rotation = (
            batch["reference_rotation"]
            if "reference_rotation" in batch and self.use_reference_rotation
            else None
        )
        if "sample_id" in batch and len(batch["sample_id"]) > 0:
            self._debug_current_sample_id = batch["sample_id"][0]
        else:
            self._debug_current_sample_id = "unknown_sample"

        poses, _ = self.normalize_camera_poses_training(
            poses,
            input_mask | target_mask,
            default_scale=pose_scale_factor,
            divison_factor=divison_factor,
        )
        with torch.inference_mode(), torch.autocast("cuda"):
            conditioning = self.prepare_conditioning(
                images,
                input_mask,
                poses,
                intrinsics,
                inlcude_latents=True,
                reference_rotation=reference_rotation,
            )

        ldm_kwargs = self.additional_ldm_conditions(
            batch, input_latents=None, is_cfg=True, num_noise_samples=num_noises_per_frame
        )

        return self.run_denoising_loop(
            input_mask,
            target_mask,
            poses,
            conditioning,
            num_inference_steps,
            num_noises_per_frame,
            init_noise=init_noise,
            **ldm_kwargs,
        )

    def debug_images(self, batch, **kwargs):
        return None
