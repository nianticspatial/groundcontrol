import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as vision_F
from omegaconf import OmegaConf

# from bitsandbytes.optim import AdamW8bit
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import osm_diffusion.training.viz as viz_utils
from osm_diffusion.semeval import DINOv1SimilarityMeasure, SegAnySimilarityMeasure
from osm_diffusion.seva import utils as seva_utils
from osm_diffusion.seva.model import SevaEncoder
from osm_diffusion.training import utils
from osm_diffusion.training.lora_utils import (
    extract_lora_state_dict,
    inject_lora_into_linears,
)
from osm_diffusion.training.loss_weighting import (
    DiffusionLossWeighting,
    EDMWeighting,
    UnitWeighting,
)
from osm_diffusion.training.pipeline_factory import create_pipeline
from osm_diffusion.training.seva_pipeline import SevaPipeline, append_dims
from osm_diffusion.training.utils import update_checkpoint_with_norm3_to_norm4

logger = logging.getLogger(__name__)


def get_lightning_module(config):
    logger.info(f"Creating SEVA lightning module...")
    # Inference YAMLs omit training/optimizer; keep an empty stub so .get() works.
    if config.get("training") is None:
        config.training = OmegaConf.create({})
    pipeline = create_pipeline(
        model_config=config.model,
        paths_config=config.paths,
    )
    module = SevaLightningModule(pipeline, config)
    if config.training.get("train_only_ldm_sat_ca", False):
        module.train_only_ldm_sat_ca()
        logger.info(f"Training only the LDM SAT CA...")
    elif config.training:
        logger.info(f"Training the full LDM according to config...")
    else:
        logger.info(f"Inference-only config (no training section).")
    return module


class SevaLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for training the SEVA model.

    This version has been refactored to call helper methods directly from the
    pipeline instance, ensuring consistency and reducing redundant code.

    Args:
        pipeline (SevaPipeline): An instance of your SEVA pipeline containing the models
                                    and helper methods for input preparation.
        cfg (Namespace or dict): A configuration object with training parameters.
    """

    def __init__(self, pipeline, cfg):
        super().__init__()
        # Store the pipeline and config
        self.pipeline: SevaPipeline = pipeline
        self.cfg = cfg

        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(cfg)

        self.loss_fn = F.mse_loss

        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none")
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
        # LPIPS requires weights to be downloaded the first time it's used
        self.val_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

        if cfg.validation.get("semeval", False):
            paths = cfg.get("paths", {})
            dino_weights = paths.get("dino_weights")
            sam_weights = paths.get("sam_weights")
            if not dino_weights or not sam_weights:
                raise ValueError(
                    "validation.semeval requires dino_weights and sam_weights in paths.yaml"
                )
            self.val_dino = DINOv1SimilarityMeasure(dino_model_weights_path=dino_weights)
            self.val_segany = SegAnySimilarityMeasure(sam_model_weights_path=sam_weights)

        # List to store validation outputs for image logging
        self.validation_step_outputs = {}
        self._train_only_ldm_sat_ca_flag = False

        self.loss_weighting: DiffusionLossWeighting = (
            EDMWeighting(sigma_data=1.0)
            if self.cfg.training.get("use_edm_weighting", False)
            else UnitWeighting()
        )
        self._lora_enabled = False
        self._lora_trainable = True
        self._lora_target_modules: list[str] = []
        self._sat_lora_enabled = False
        self._sat_lora_trainable = True
        self._sat_lora_target_modules: list[str] = []
        self._maybe_enable_lora()

        if self.cfg.model.get("train_sat_encoder", False):
            self.pipeline.sat_encoder.train()
            self.pipeline.sat_encoder.requires_grad_(True)

        self.train_sat_dpt_head = self.cfg.model.get("train_sat_dpt_head", False)
        if self.train_sat_dpt_head and hasattr(self.pipeline, "sat_encoder"):
            setter = getattr(self.pipeline.sat_encoder, "set_dpt_head_trainable", None)
            if setter is not None:
                setter(True)
            else:
                logger.warning("train_sat_dpt_head enabled but sat encoder has no DPT head.")
        elif self.train_sat_dpt_head:
            logger.warning("train_sat_dpt_head enabled but no sat encoder found!")

        self.save_validation_results = self.cfg.validation.get("save_validation_results", False)
        self.save_validation_camera_specs = self.cfg.validation.get(
            "save_validation_camera_specs", False
        )
        self.validation_debug_dir = self.cfg.validation.get("validation_debug_dir", None)
        self.validation_scale_factors = self.cfg.validation.get("validation_scale_factors", [2.0])
        self.save_validation_grid_images = self.cfg.validation.get(
            "save_validation_grid_images", True
        )
        self._eval_dataset_names = self.cfg.get("val_data", self.cfg.data).datasets

        self.save_images_dir = None
        self.garbage_collection_after_validation_step = False

        if self.cfg.training.get("use_distillation_loss", False):
            kwargs = {}
            if self.cfg.model.name.endswith("controlnet"):
                # kwargs = {
                #     "decoder_blocks_with_txr": list(range(len(self.pipeline.ldm.output_blocks)))
                # }
                self.ldm_encoder = self.pipeline.ldm
            else:
                self.ldm_encoder = SevaEncoder(
                    self.pipeline.ldm.params,
                    run_decoder=self.cfg.model.get("sat_block_in_decoder", False),
                    **kwargs,
                )
            state_dict = seva_utils.get_model_state_dict(
                model_version=1.1,
                pretrained_model_name_or_path=self.cfg.paths.pretrained_weights_path,
                weight_name="model.safetensors",
                device="cpu",  # Load to CPU first
            )
            self.ldm_encoder.load_state_dict(state_dict, strict=False)
            self.ldm_encoder.eval()
            self.ldm_encoder.requires_grad_(False)
            self.ldm_encoder.to(self.device)

    def _should_apply_lora_for_seva(self, module_name: str) -> bool:
        """For base SEVA model: target linears matching target_modules anywhere in LDM."""
        if self._lora_target_modules and not any(
            module_name.endswith(target_name) for target_name in self._lora_target_modules
        ):
            return False
        return True

    def _should_apply_lora_to_satellite_blocks(self, module_name: str) -> bool:
        in_satellite_branch = ".satellite_blocks." in module_name or ".height_cond_blocks." in module_name
        if not in_satellite_branch:
            return False
        if self._sat_lora_target_modules and not any(
            module_name.endswith(target_name) for target_name in self._sat_lora_target_modules
        ):
            return False
        return True

    def _maybe_enable_lora(self) -> None:
        lora_cfg = self.cfg.training.get("lora", {})
        if not lora_cfg.get("enabled", False):
            return

        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 8.0))
        dropout = float(lora_cfg.get("dropout", 0.0))
        self._lora_trainable = bool(lora_cfg.get("trainable", True))
        self._lora_target_modules = [str(x) for x in lora_cfg.get("target_modules", [])]

        replaced = inject_lora_into_linears(
            self.pipeline.ldm,
            should_wrap=self._should_apply_lora_for_seva,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        if replaced == 0:
            logger.warning("LoRA was enabled but no eligible linear layers were found.")
            return

        self._lora_enabled = True
        logger.info(
            "Enabled LoRA (rank=%s alpha=%s dropout=%s trainable=%s targets=%s), replaced %s linear layers.",
            rank,
            alpha,
            dropout,
            self._lora_trainable,
            self._lora_target_modules if self._lora_target_modules else "all-linear",
            replaced,
        )

        sat_lora_cfg = lora_cfg.get("satellite", {})
        if sat_lora_cfg.get("enabled", False):
            sat_rank = int(sat_lora_cfg.get("rank", rank))
            sat_alpha = float(sat_lora_cfg.get("alpha", alpha))
            sat_dropout = float(sat_lora_cfg.get("dropout", dropout))
            self._sat_lora_trainable = bool(sat_lora_cfg.get("trainable", True))
            self._sat_lora_target_modules = [
                str(x) for x in sat_lora_cfg.get("target_modules", [])
            ]
            sat_replaced = inject_lora_into_linears(
                self.pipeline.ldm,
                should_wrap=self._should_apply_lora_to_satellite_blocks,
                rank=sat_rank,
                alpha=sat_alpha,
                dropout=sat_dropout,
            )
            if sat_replaced == 0:
                logger.warning(
                    "Satellite LoRA was enabled but no eligible satellite linear layers were found."
                )
            else:
                self._sat_lora_enabled = True
                logger.info(
                    "Enabled satellite LoRA (rank=%s alpha=%s dropout=%s trainable=%s targets=%s), replaced %s satellite linear layers.",
                    sat_rank,
                    sat_alpha,
                    sat_dropout,
                    self._sat_lora_trainable,
                    self._sat_lora_target_modules
                    if self._sat_lora_target_modules
                    else "all-satellite-linears",
                    sat_replaced,
                )

    def compute_distillation_loss(self, intermeds):
        clean_processed_latents = intermeds["clean_processed_latents"]
        conditioning = intermeds["conditioning"]

        distill_loss = 0.0
        with torch.no_grad():
            _, teacher_intermeds = self.pipeline.ldm(
                clean_processed_latents,
                t=torch.zeros(
                    clean_processed_latents.shape[0],
                    device=clean_processed_latents.device,
                    dtype=torch.long,
                ),
                y=conditioning["cross_attn_cond"],
                dense_y=conditioning["plucker_coords"],
                num_frames=conditioning.get("num_frames", None),
                disable_sat_cond=True,
            )
            for k, val in teacher_intermeds.items():
                for teacher_embed, student_embed in zip(val, intermeds[k]):
                    distill_loss += F.mse_loss(student_embed, teacher_embed)
        return distill_loss

    def compute_loss(self, model_output, target, sigmas_t, target_mask, intermeds=None):
        w = self.loss_weighting(sigmas_t)
        w = append_dims(w, model_output.ndim)
        loss_type = self.cfg.training.get("loss_type", "l2")
        if loss_type == "l2":
            loss = w * (model_output - target) ** 2
        elif loss_type == "l1":
            loss = w * (model_output - target).abs()
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        # if intermeds and self.cfg.training.get("use_distillation_loss", False):
        #     distill_loss = self.compute_distillation_loss(intermeds)
        #     print("Distillation loss: ", distill_loss.item())
        #     loss = loss + self.cfg.training.get("distillation_loss_weight", 1e-2) * distill_loss
        loss = torch.mean(loss[target_mask])
        return loss

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Log pre-clip gradient stats, then apply standard Lightning clipping."""
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return

        grads = [
            p.grad.detach()
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grads:
            return

        grad_sq_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        for g in grads:
            grad_sq_sum = grad_sq_sum + g.float().pow(2).sum()
        pre_clip_norm = torch.sqrt(grad_sq_sum)

        clip_val_tensor = torch.tensor(float(gradient_clip_val), device=pre_clip_norm.device)
        was_clipped = (pre_clip_norm > clip_val_tensor).float()
        clip_coef = torch.minimum(
            torch.ones_like(pre_clip_norm), clip_val_tensor / (pre_clip_norm + 1e-12)
        )

        self.log("grad/pre_clip_norm", pre_clip_norm, on_step=True, on_epoch=False, logger=True)
        self.log("grad/was_clipped", was_clipped, on_step=True, on_epoch=False, logger=True)
        self.log("grad/clip_coef", clip_coef, on_step=True, on_epoch=False, logger=True)

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def _is_trainable_layer(self, layer_name: str) -> bool:
        other_trainable_params = self.cfg.training.get("other_params_to_train", [])

        if self._lora_enabled:
            if "lora_A" in layer_name or "lora_B" in layer_name:
                if ".satellite_blocks." in layer_name or ".height_cond_blocks." in layer_name:
                    return self._sat_lora_trainable
                return self._lora_trainable
            if "zero_convs" in layer_name or "middle_block_output" in layer_name:
                return True
            if ".satellite_blocks." in layer_name or ".height_cond_blocks." in layer_name:
                return True
            if any(param in layer_name for param in other_trainable_params):
                return True
            return False

        trainable_blocks = self.cfg.training.get(
            "transformer_blocks_to_train", ["transformer_blocks"]
        )
        trainable_layers = self.cfg.training.transformer_layers_to_train
        if any(block in layer_name for block in trainable_blocks) and any(
            layer in layer_name for layer in trainable_layers
        ):
            if self.cfg.model.sat_ca_input_only:
                if "input_block" not in layer_name and "middle_block" not in layer_name:
                    return False
            return True
        if any(param in layer_name for param in other_trainable_params):
            return True
        return False

    def train_only_ldm_sat_ca(self):
        for param in self.pipeline.ldm.parameters():
            param.requires_grad = False
        all_params_frozen = True

        for name, param in self.pipeline.ldm.named_parameters():
            if self._is_trainable_layer(name):
                param.requires_grad = True
                all_params_frozen = False

        if all_params_frozen:
            logger.warning("All parameters in the LDM are frozen, hope you know what youre doing.")

        self._train_only_ldm_sat_ca_flag = True

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Called when saving a checkpoint. We customize the model state_dict here.
        Note: PyTorch Lightning will automatically save callback states (including EMA).
        """
        # Replace the full pipeline state_dict with just the ldm model
        # This reduces checkpoint size while preserving callback states
        checkpoint["state_dict"] = self.pipeline.ldm.state_dict()
        lora_state_dict = extract_lora_state_dict(checkpoint["state_dict"])
        if lora_state_dict:
            checkpoint["lora_state_dict"] = lora_state_dict
            checkpoint["lora_config"] = {
                "enabled": bool(self.cfg.training.get("lora", {}).get("enabled", False)),
                "rank": int(self.cfg.training.get("lora", {}).get("rank", 8)),
                "alpha": float(self.cfg.training.get("lora", {}).get("alpha", 8.0)),
                "dropout": float(self.cfg.training.get("lora", {}).get("dropout", 0.0)),
                "trainable": bool(self.cfg.training.get("lora", {}).get("trainable", True)),
                "target_modules": list(self.cfg.training.get("lora", {}).get("target_modules", [])),
                "satellite": {
                    "enabled": bool(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("enabled", False)
                    ),
                    "rank": int(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("rank", 8)
                    ),
                    "alpha": float(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("alpha", 8.0)
                    ),
                    "dropout": float(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("dropout", 0.0)
                    ),
                    "trainable": bool(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("trainable", True)
                    ),
                    "target_modules": list(
                        self.cfg.training.get("lora", {}).get("satellite", {}).get("target_modules", [])
                    ),
                },
            }

        if self.cfg.model.get("train_sat_encoder", False) or self.train_sat_dpt_head:
            if hasattr(self.pipeline, "sat_encoder"):
                checkpoint["sat_encoder"] = self.pipeline.sat_encoder.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Called when loading a checkpoint. We define how to load our custom state here.
        This version is backward-compatible and can load old, full checkpoints.
        """
        if "state_dict" not in checkpoint and "lora_state_dict" in checkpoint:
            logger.info("Loading adapter-only LoRA checkpoint.")
            checkpoint["state_dict"] = checkpoint["lora_state_dict"]

        full_state_dict = checkpoint["state_dict"]

        # If both are present, use adapter weights as source of truth for LoRA tensors.
        if "lora_state_dict" in checkpoint:
            full_state_dict.update(checkpoint["lora_state_dict"])

        # This is to maintain compatibility with old checkpoints,
        # which saved the full state_dict of the pipeline.
        prefix = "pipeline.ldm."
        is_full_checkpoint = False
        for k, v in full_state_dict.items():
            if k.startswith(prefix):
                is_full_checkpoint = True
                break

        if not is_full_checkpoint:
            new_full_state_dict = self.state_dict()
            ldm_weights = {f"pipeline.ldm.{k}": v for k, v in checkpoint["state_dict"].items()}
            if self.cfg.training.get("init_norm4_from_norm3", False):
                ldm_weights = update_checkpoint_with_norm3_to_norm4(ldm_weights)
            new_full_state_dict.update(ldm_weights)
            checkpoint["state_dict"] = new_full_state_dict
        else:
            if self.cfg.training.get("init_norm4_from_norm3", False):
                checkpoint["state_dict"] = update_checkpoint_with_norm3_to_norm4(
                    checkpoint["state_dict"]
                )

        # Optionally load satellite encoder weights, if we are training it
        if (
            self.cfg.model.get("train_sat_encoder", False) or self.train_sat_dpt_head
        ) and "sat_encoder" in checkpoint:
            print("Sat encoder weights found in checkpoint, loading...")
            sat_encoder_weights = {
                f"pipeline.sat_encoder.{k}": v for k, v in checkpoint["sat_encoder"].items()
            }
            checkpoint["state_dict"].update(sat_encoder_weights)

    def log_train_metrics(
        self,
        epsilon_pred,
        sigmas_t,
        noisy_latents,
        target_mask,
        images,
        input_mask,
        prefix="",
    ):
        """Unused function to log training metrics like PSNR and SSIM."""
        with torch.no_grad():
            # 1. Predict the clean latents (x0) from the model's noise prediction
            pred_x0_latents = self.pipeline.convert_model_output_to_x0(
                epsilon_pred, sigmas_t.view(-1, 1, 1, 1, 1), noisy_latents
            )

            # 2. Decode the predicted clean latents into image space
            # We only need to decode the target views for metric calculation
            B, N = pred_x0_latents.shape[:2]
            decoded_images = self.pipeline.vae.decode(
                pred_x0_latents.view(-1, *pred_x0_latents.shape[2:])
            )
            decoded_images = torch.clamp(decoded_images, -1, 1)
            decoded_images = decoded_images.view(B, N, *decoded_images.shape[1:])

            # 4. Denormalize both to [0, 1] range
            decoded_images = (decoded_images + 1) / 2.0
            gt_images_norm = (images + 1) / 2.0

            # 3. Get ground truth target images
            decoded_images_flat = decoded_images[target_mask]
            gt_images_norm_flat = gt_images_norm[target_mask]

            # 5. Update and log metrics
            self.train_psnr.update(decoded_images_flat, gt_images_norm_flat)
            self.train_ssim.update(decoded_images_flat, gt_images_norm_flat)

            self.log(
                f"train/{prefix}psnr_step",
                self.train_psnr,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                f"train/{prefix}ssim_step",
                self.train_ssim,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

    @torch.no_grad()
    def _base_model_forward(self, model_input, sigmas_t, conditioning, timesteps, num_frames):
        concat_reshaped = conditioning["concat_condition"].view(
            -1, num_frames, *conditioning["concat_condition"].shape[2:]
        )
        scaled_conditioned_input = self.pipeline.scale_and_camera_condition_input(
            model_input, sigmas_t, concat_reshaped
        )
        scaled_conditioned_input = scaled_conditioned_input.view(
            -1, *scaled_conditioned_input.shape[2:]
        )

        _, intermeds = self.ldm_encoder(
            scaled_conditioned_input,
            t=timesteps,
            y=conditioning["cross_attn_cond"],
            dense_y=conditioning["plucker_coords"],
            num_frames=num_frames,
        )
        return intermeds

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step. PyTorch Lightning handles moving the batch
        to the correct device.
        """
        # 1. Prepare data from batch
        images = batch["images"]
        input_mask = batch["input_mask"]
        target_mask = batch["target_mask"]
        poses = batch["camera_poses"]
        intrinsics = batch["intrinsics"]
        reference_rotation = (
            batch["reference_rotation"]
            if "reference_rotation" in batch and self.pipeline.use_reference_rotation
            else None
        )

        B, N, C, H, W = images.shape

        # 2. Use the pipeline to prepare all conditioning signals.
        with torch.no_grad():
            if self.hparams.training.get("normalize_camera_poses", True):
                poses, _ = self.pipeline.normalize_camera_poses_training(
                    poses,
                    input_mask | target_mask,
                    divison_factor=self.hparams.training.get("divison_factor"),
                )

            conditioning = self.pipeline.prepare_conditioning(
                images,
                input_mask,
                poses,
                intrinsics,
                inlcude_latents=False,
                reference_rotation=reference_rotation,
            )
            uncond_cross_attn_cond, uncond_concat_condition, uncond_plucker_coords = (
                self.pipeline.init_unconditionional(
                    conditioning["cross_attn_cond"],
                    conditioning["concat_condition"],
                    conditioning["plucker_coords"],
                )
            )
            uncond_input_mask = torch.bernoulli(
                torch.ones((B,), device=self.device) * self.hparams.training.uncond_drop_prob
            ).bool()
            conditioning["cross_attn_cond"][uncond_input_mask] = uncond_cross_attn_cond[
                uncond_input_mask
            ]
            conditioning["concat_condition"][uncond_input_mask] = uncond_concat_condition[
                uncond_input_mask
            ]
            image_uncond_mask = torch.bernoulli(
                torch.ones((B,), device=self.device)
                * self.hparams.training.get("image_uncond_drop_prob", 0.0)
            ).bool()

            clean_latents = self.pipeline.encode_images(images)

        # 3. Sample noise and timesteps
        noise = torch.randn_like(clean_latents)
        if not self.training and "debug_ts" in batch:
            timesteps = batch["debug_ts"].to(self.device).reshape(-1)
        else:
            timesteps = torch.randint(
                0,
                self.pipeline.scheduler.config.num_train_timesteps,
                (B,),
                device=self.device,
            )
        if self.cfg.training.get("timesteps_cubic_sampling", False):
            timesteps = (
                1 - (timesteps / self.pipeline.scheduler.config.num_train_timesteps) ** 3
            ) * self.pipeline.scheduler.config.num_train_timesteps
            timesteps = (
                timesteps.int()
                .clamp(0, self.pipeline.scheduler.config.num_train_timesteps - 1)
                .to(self.device)
            )

        # 4. Add noise to clean latents to create the model input `x_t`
        noisy_latents = self.pipeline.scheduler.add_noise(clean_latents, noise, timesteps)
        sigmas_t = self.pipeline.scheduler.train_sigmas[timesteps].to(self.device)

        # 5. Prepare the final model input
        model_input = noisy_latents.clone()
        replace_mask = (~uncond_input_mask[:, None]) & (~image_uncond_mask[:, None]) & input_mask
        # Use torch.where instead of inplace assignment to avoid inference tensor issues
        model_input[replace_mask] = clean_latents[replace_mask].to(model_input)

        scaled_conditioned_input = self.pipeline.scale_and_camera_condition_input(
            model_input, sigmas_t, conditioning["concat_condition"]
        )

        additional_ldm_kwargs = self.pipeline.additional_ldm_conditions(
            batch, scaled_conditioned_input, uncond_input_mask=uncond_input_mask
        )
        (
            scaled_conditioned_input,
            updated_ldm_kwargs,
            num_frames,
            conditioning["cross_attn_cond"],
            conditioning["plucker_coords"],
        ) = self.pipeline.update_inputs_ldm_kwargs_for_timestep(
            scaled_conditioned_input,
            additional_ldm_kwargs,
            sigmas_t,
            N,
            conditioning["cross_attn_cond"],
            conditioning["plucker_coords"],
            uncond_input_mask=uncond_input_mask,
        )

        (
            scaled_conditioned_input,
            conditioning["cross_attn_cond"],
            conditioning["plucker_coords"],
            timesteps,
        ) = self.pipeline.reshape_model_inputs(
            scaled_conditioned_input,
            conditioning["plucker_coords"],
            conditioning["cross_attn_cond"],
            timesteps,
        )

        # print('timesteps', timesteps.detach().cpu().numpy().tolist(), sigmas_t.detach().cpu().numpy().tolist())
        # 6. Run the LDM model with the exact same signature as inference
        epsilon_pred = self.pipeline.ldm(
            scaled_conditioned_input,
            t=timesteps,
            y=conditioning["cross_attn_cond"],
            dense_y=conditioning["plucker_coords"],
            num_frames=num_frames,
            **updated_ldm_kwargs,
        )
        if not isinstance(epsilon_pred, torch.Tensor):
            epsilon_pred, intermeds = epsilon_pred
        else:
            intermeds = None
        epsilon_pred, aux_loss = self.pipeline.update_epsilon_pred_for_timestep(
            epsilon_pred,
            num_frames,
        )
        epsilon_pred = epsilon_pred.view(B, N, *epsilon_pred.shape[1:])

        # 7. Compute loss only on the target views
        loss = self.compute_loss(
            epsilon_pred,
            noise,
            sigmas_t,
            target_mask | ((uncond_input_mask[:, None] | image_uncond_mask[:, None]) & input_mask),
            intermeds=None,
        )

        if self.cfg.training.get("use_distillation_loss", False):
            assert intermeds is not None, "Intermeds are not available"
            assert "satellite_images" in batch, "Satellite images are not available"
            model_input_with_support_views = noisy_latents.clone()
            valid_satellite_mask = (
                torch.abs(batch["satellite_images"].reshape(B, -1)) > 0.01
            ).any(dim=1)
            # we drop image unconditional here, because its good to have this loss in image_uncond.
            support_mask = (~uncond_input_mask[:, None]) & (
                input_mask | (valid_satellite_mask[:, None] & batch["support_mask"])
            )

            model_input_with_support_views[support_mask] = clean_latents[support_mask].to(
                model_input_with_support_views
            )
            support_intermeds = self._base_model_forward(
                model_input_with_support_views, sigmas_t, conditioning, timesteps, num_frames
            )
            intermed_losses = []
            weight = (1 - support_mask.float()).reshape(-1)[:, None, None, None]
            for student_intermed, teacher_intermed in zip(intermeds, support_intermeds):
                intermed_losses.append(
                    F.mse_loss(
                        student_intermed,
                        teacher_intermed,
                        weight=weight.expand_as(student_intermed),
                    )
                )
            intermed_loss = torch.mean(torch.stack(intermed_losses))
            loss = loss + self.cfg.training.get("distillation_loss_weight", 1.0) * intermed_loss

            # weight = (1 - support_mask.float()).reshape(-1)
            # for student_intermed, teacher_intermed in zip(intermeds, support_intermeds):
            #     feature_loss = F.mse_loss(
            #         student_intermed,
            #         teacher_intermed,
            #         reduction="none",
            #     ).mean(dim=(-1, -2, -3))
            #     feature_loss = (weight * feature_loss).reshape(B, -1).sum(dim=1) / (
            #         weight.reshape(B, -1).sum(dim=1) + 1e-6
            #     )
            #     intermed_losses.append(torch.mean(feature_loss))

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        loss = loss + 0.0 * aux_loss
        return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        """
        # Create the optimizer
        optim_class = AdamW
        if self._train_only_ldm_sat_ca_flag:
            print("Configuring optimizer for training only the LDM SAT CA...")
            requires_grad_params = [
                param for name, param in self.pipeline.ldm.named_parameters() if param.requires_grad
            ]
            trainable_params = [
                (name, param)
                for name, param in self.pipeline.ldm.named_parameters()
                if self._is_trainable_layer(name) and param.requires_grad
            ]
            print(
                "trainable_params / requires_grad_params",
                len(trainable_params),
                len(requires_grad_params),
            )
            trainable_params_names = [p[0] for p in trainable_params]
            trainable_params = [p[1] for p in trainable_params]
            for name, param in self.pipeline.ldm.named_parameters():
                if name in trainable_params_names:
                    print(name, param.requires_grad, self._is_trainable_layer(name))
                else:
                    assert not param.requires_grad, f"Parameter {name} is not trainable"
            ldm_param_list = trainable_params
        else:
            print("Training the full LDM...")
            ldm_param_list = list(self.pipeline.ldm.parameters())

        param_groups = [{"params": ldm_param_list}]

        sat_params = []
        if self.cfg.model.get("train_sat_encoder", False) and hasattr(self.pipeline, "sat_encoder"):
            sat_params = [p for p in self.pipeline.sat_encoder.parameters() if p.requires_grad]

        dpt_params = []
        if (
            self.train_sat_dpt_head
            and hasattr(self.pipeline, "sat_encoder")
            and hasattr(self.pipeline.sat_encoder, "get_dpt_head_parameters")
        ):
            dpt_params = [
                p for p in self.pipeline.sat_encoder.get_dpt_head_parameters() if p.requires_grad
            ]
            if sat_params:
                dpt_param_ids = {id(p) for p in dpt_params}
                sat_params = [p for p in sat_params if id(p) not in dpt_param_ids]

        if sat_params:
            sat_lr = self.cfg.optimizer.get("sat_encoder_learning_rate")
            if sat_lr is not None:
                print("Training the SAT encoder with learning rate: ", sat_lr)
                param_groups.append({"params": sat_params, "lr": sat_lr})
            else:
                print("Training the SAT encoder with default learning rate")
                param_groups[0]["params"].extend(sat_params)

        if dpt_params:
            dpt_lr = self.cfg.optimizer.get(
                "dpt_head_learning_rate",
                self.cfg.optimizer.get(
                    "sat_encoder_learning_rate", self.hparams.optimizer.learning_rate
                ),
            )
            print("Training the DPT head with learning rate: ", dpt_lr)
            param_groups.append({"params": dpt_params, "lr": dpt_lr})

        optimizer = optim_class(
            param_groups,
            lr=self.hparams.optimizer.learning_rate,
            betas=(
                self.hparams.optimizer.adam_beta1,
                self.hparams.optimizer.adam_beta2,
            ),
            weight_decay=self.hparams.optimizer.adam_weight_decay,
            eps=self.hparams.optimizer.adam_epsilon,
        )

        # Create the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.hparams.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.training.lr_warmup_steps,
            num_training_steps=self.hparams.training.max_train_steps,  # Assumes this is in your config
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def validate_and_get_metrics(self, batch, scale_factor, ds_name, return_attn=False):
        # 1. Get generated images... (logic is the same)
        self.pipeline.return_attn = return_attn
        generated_images, all_debug_dicts, _ = self.pipeline(
            batch,
            num_inference_steps=self.hparams.validation.get("denoising_steps", 50),
            pose_scale_factor=scale_factor,
            divison_factor=self.hparams.training.get("divison_factor"),
        )
        self.pipeline.return_attn = False
        if all_debug_dicts and not self.save_validation_results:
            debug_logs = utils.get_debug_attn_logs(all_debug_dicts, ds_name)
            for k, v in debug_logs.items():
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=False)
            # Immediately clear debug dicts to free CPU memory
            del debug_logs
        # Always clear all_debug_dicts to free memory, even if empty
        if all_debug_dicts:
            # Clear all tensors in debug dicts - need to properly delete references
            for debug_dict in all_debug_dicts:
                for key in list(debug_dict.keys()):
                    val = debug_dict.pop(key)
                    if isinstance(val, torch.Tensor):
                        del val
                debug_dict.clear()
            all_debug_dicts.clear()
        del all_debug_dicts

        # 2. Get ground truth... (logic is the same)
        gt_images = batch["images"]
        target_mask = batch["target_mask"]

        # 3. Select target views... (logic is the same)
        gt_target_images = gt_images[target_mask].float()
        generated_target_images = generated_images[target_mask].float()

        # 4. Denormalize images... (logic is the same)
        gt_target_images_norm = (gt_target_images + 1) / 2.0
        generated_target_images_norm = (generated_target_images + 1) / 2.0

        # 5. Calculate metrics (logic is the same)
        psnr = self.val_psnr(generated_target_images_norm, gt_target_images_norm)
        ssim = self.val_ssim(generated_target_images_norm, gt_target_images_norm)
        # Torch lpips expects images in the range [-1, 1]
        lpips = self.val_lpips(generated_target_images, gt_target_images)

        if self.cfg.validation.get("semeval", False):
            # Update metrics with current batch
            self.val_dino.update(generated_target_images_norm, gt_target_images_norm)
            self.val_segany.update(generated_target_images_norm, gt_target_images_norm)
            # Compute metrics immediately to get per-batch value
            semeval_dino = self.val_dino.compute()
            semeval_segany = self.val_segany.compute()
            # CRITICAL: Reset metric state to prevent CPU memory accumulation across batches
            # These metrics accumulate CPU tensors in self.preds and self.target lists
            self.val_dino.reset()
            self.val_segany.reset()
        else:
            semeval_dino = None
            semeval_segany = None

        return generated_images, psnr, ssim, lpips, semeval_dino, semeval_segany

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Performs a single validation step.
        """
        # Create a list of names for your validation datasets
        current_dataset_name = self._eval_dataset_names[dataloader_idx]
        if current_dataset_name not in self.validation_step_outputs:
            self.validation_step_outputs[current_dataset_name] = []

        # For each scale factor, compute metrics and log them
        all_generated_images = []
        gt_images = batch["images"]
        all_psnr = []
        all_ssim = []
        all_lpips = []
        all_semeval_dino = []
        all_semeval_segany = []

        for scale_factor in self.validation_scale_factors:
            generated_images, psnr, ssim, lpips, semeval_dino, semeval_segany = (
                self.validate_and_get_metrics(
                    batch, scale_factor, current_dataset_name, return_attn=False
                )
            )
            all_generated_images.append(generated_images)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)
            all_semeval_dino.append(semeval_dino)
            all_semeval_segany.append(semeval_segany)

        # Choose best scale factor based on PSNR
        best_idx = torch.argmax(torch.stack(all_psnr), dim=0)
        generated_images = all_generated_images[best_idx]
        psnr = all_psnr[best_idx]
        ssim = all_ssim[best_idx]
        lpips = all_lpips[best_idx]
        semeval_dino = all_semeval_dino[best_idx]
        semeval_segany = all_semeval_segany[best_idx]
        best_scale = self.validation_scale_factors[best_idx]

        # Clean up accumulated lists to free memory
        # Only keep the best one, clear the rest
        for idx, img in enumerate(all_generated_images):
            if idx != best_idx:
                del img
        del (
            all_generated_images,
            all_psnr,
            all_ssim,
            all_lpips,
            all_semeval_dino,
            all_semeval_segany,
        )

        # Log the metrics with a dataset-specific prefix
        self.log(
            f"val/{current_dataset_name}/psnr",
            torch.mean(psnr),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{current_dataset_name}/ssim",
            torch.mean(ssim),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{current_dataset_name}/lpips",
            lpips,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if self.cfg.validation.get("semeval", False):
            self.log(
                f"val/{current_dataset_name}/semeval_dino",
                semeval_dino,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/{current_dataset_name}/semeval_segany",
                semeval_segany,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        if self.save_validation_results:
            for b in range(batch["images"].shape[0]):
                # Sample ID is a tuple of various identifiers, we join them with '__' to form a unique string.
                # only 40 chars to avoid too long paths
                sample_id = "__".join([str(s)[:50] for s in batch["sample_id"][b]])
                debug_dir = Path(self.validation_debug_dir) / current_dataset_name / sample_id
                debug_dir.mkdir(parents=True, exist_ok=True)
                gt_images_normd = gt_images[b][batch["input_mask"][b]] / 2.0 + 0.5
                for i in range(gt_images_normd.shape[0]):
                    save_path = debug_dir / f"input_{i}_gt.png"
                    torchvision.utils.save_image(
                        gt_images_normd[i],
                        save_path,
                    )
                generated_images_b = generated_images[b][batch["target_mask"][b]] / 2.0 + 0.5
                gt_images_b = gt_images[b][batch["target_mask"][b]] / 2.0 + 0.5
                satellite_image = batch["satellite_images"][b, 0] / 2.0 + 0.5
                for i in range(generated_images_b.shape[0]):
                    save_path = debug_dir / f"target_{i}_gen.png"
                    torchvision.utils.save_image(
                        generated_images_b[i],
                        save_path,
                    )
                    save_path = debug_dir / f"target_{i}_gt.png"
                    torchvision.utils.save_image(
                        gt_images_b[i],
                        save_path,
                    )
                sat_save_path = debug_dir / f"satellite_image.png"
                torchvision.utils.save_image(
                    satellite_image,
                    sat_save_path,
                )
                metrics = {
                    "psnr": psnr.float().cpu().numpy().tolist(),
                    "ssim": ssim.float().cpu().numpy().tolist(),
                    "lpips": lpips.float().cpu().numpy().tolist(),
                    "best_scale_factor": best_scale,
                }
                metrics_path = debug_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                if self.save_validation_camera_specs:
                    target_indices = torch.nonzero(batch["target_mask"][b], as_tuple=False).squeeze(
                        -1
                    )
                    target_intrinsics = (
                        batch["intrinsics"][b][target_indices].detach().cpu().numpy()
                    )
                    target_poses = batch["camera_poses"][b][target_indices].detach().cpu().numpy()
                    height, width = gt_images.shape[-2], gt_images.shape[-1]
                    camera_spec = {
                        "sample_id": sample_id,
                        "dataset": current_dataset_name,
                        "image_resolution": [int(width), int(height)],
                        "target_views": [],
                    }
                    for view_idx, pose, intr in zip(
                        target_indices.tolist(), target_poses, target_intrinsics
                    ):
                        camera_spec["target_views"].append(
                            {
                                "index": int(view_idx),
                                "pose": pose.tolist(),
                                "intrinsics": {
                                    "fx": float(intr[0][0]),
                                    "fy": float(intr[1][1]),
                                    "cx": float(intr[0][2]),
                                    "cy": float(intr[1][2]),
                                },
                                "image_files": {
                                    "gen": f"target_{len(camera_spec['target_views'])}_gen.png",
                                    "gt": f"target_{len(camera_spec['target_views'])}_gt.png",
                                },
                            }
                        )
                    camera_spec_path = debug_dir / "target_camera_spec.json"
                    with open(camera_spec_path, "w") as f:
                        json.dump(camera_spec, f, indent=4)

            satellite_debug_images = self.pipeline.debug_images(batch)
            if satellite_debug_images is not None:
                satellite_debug_images = satellite_debug_images / 2.0 + 0.5
                input_satellite_debug_images = satellite_debug_images[batch["input_mask"][0]]
                for i in range(input_satellite_debug_images.shape[0]):
                    save_path = debug_dir / f"sampling_input_{i}.png"
                    torchvision.utils.save_image(
                        input_satellite_debug_images[i],
                        save_path,
                    )
                target_satellite_debug_images = satellite_debug_images[batch["target_mask"][0]]
                for i in range(target_satellite_debug_images.shape[0]):
                    save_path = debug_dir / f"sampling_target_{i}.png"
                    torchvision.utils.save_image(
                        target_satellite_debug_images[i],
                        save_path,
                    )

        # Collect data for the first 5 batches to create debug images
        if batch_idx < 5 and self.save_validation_grid_images:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                satellite_debug_images = self.pipeline.debug_images(batch)

            # We only need the first sample from each of these batches
            output_data = [
                batch["images"][0][batch["input_mask"][0]],
                generated_images[0][batch["target_mask"][0]],
                batch["images"][0][batch["target_mask"][0]],
            ]
            if satellite_debug_images is not None:
                satellite_debug_images = satellite_debug_images[batch["target_mask"][0]]
                satellite_debug_images = vision_F.resize(
                    satellite_debug_images,
                    batch["images"].shape[-1],
                )
                output_data = output_data + [satellite_debug_images]

            self.validation_step_outputs[current_dataset_name].append(output_data)

        if self.garbage_collection_after_validation_step:
            # Force garbage collection after each validation step to prevent memory accumulation
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to log debug images.
        """
        if not self.validation_step_outputs:
            return
        if not self.save_images_dir:
            if not self.logger or not self.logger.experiment:
                logger.warning("No logger found, skipping validation image logging.")
                return
        else:
            logger.info(f"Saving validation images to {self.save_images_dir}")

        for dataset_name, output_data in self.validation_step_outputs.items():
            for i, output_data in enumerate(output_data):
                debug_grid = viz_utils.create_debug_grid(output_data, normalize=True)
                if self.save_images_dir:
                    save_path = Path(self.save_images_dir, f"{dataset_name}__debug_grid_{i}.png")
                    torchvision.utils.save_image(debug_grid, save_path, normalize=False)
                    print(f"Saved validation images to {save_path}")

                if self.logger and self.logger.experiment:
                    self.logger.experiment.add_image(
                        f"val/{dataset_name}/sample_grid_{i}", debug_grid, self.global_step
                    )

        # Clear the list to free up memory
        self.validation_step_outputs.clear()
