import torch
import torch.nn.functional as F

from osm_diffusion.training.seva_pipeline import SevaPipeline


class SatelliteSevaPipeline(SevaPipeline):
    """
    An intermediate pipeline that extends SevaPipeline to support an optional
    dual classifier-free guidance (CFG) mode.
    """

    def __init__(
        self,
        *args,
        use_dual_cfg: bool = False,
        cfg_scale_image: float = 2.0,
        cfg_scale_sat: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_dual_cfg = use_dual_cfg
        self.cfg_scale_image = cfg_scale_image
        self.cfg_scale_sat = cfg_scale_sat

        if self.use_dual_cfg:
            print(
                f"INFO: Pipeline initialized in DUAL CFG mode. Image CFG: {self.cfg_scale_image}, Sat CFG: {self.cfg_scale_sat}"
            )
        else:
            print(f"INFO: Pipeline initialized in SINGLE CFG mode. CFG Scale: {self.cfg_scale}")

    def prepare_inputs_cfg(
        self, model_input, noisy_latents, concat_condn, plucker_coords, cross_attn_condn
    ):
        """Overrides base method to conditionally prepare inputs for single or dual CFG."""
        if self.use_dual_cfg:
            # 3-way batch: [full_cond, image_only_cond, uncond]
            uncond_cross_attn = torch.zeros_like(cross_attn_condn)
            uncond_concat = torch.cat(
                [torch.zeros_like(concat_condn[:, :, :1]), concat_condn[:, :, 1:]],
                dim=2,
            )
            model_input_batch = torch.cat([model_input, model_input, noisy_latents], dim=0)
            cross_attn_condn_batch = torch.cat(
                [cross_attn_condn, cross_attn_condn, uncond_cross_attn], dim=0
            )
            concat_condn_batch = torch.cat([concat_condn, concat_condn, uncond_concat], dim=0)
            dense_vector_condn_batch = torch.cat([plucker_coords] * 3, dim=0)
            return (
                model_input_batch,
                cross_attn_condn_batch,
                concat_condn_batch,
                dense_vector_condn_batch,
            )
        else:
            # Fallback to original 2-way batch logic from SevaPipeline
            return super().prepare_inputs_cfg(
                model_input, noisy_latents, concat_condn, plucker_coords, cross_attn_condn
            )

    def get_output_cfg(self, predicted_output, cfg_scale):
        """Overrides base method to conditionally apply single or dual CFG formula."""
        if self.use_dual_cfg:
            # Apply dual CFG formula
            sat_cond_pred, image_cond_pred, uncond_pred = predicted_output.chunk(3, dim=0)
            image_guidance = self.cfg_scale_image * (image_cond_pred - uncond_pred)
            satellite_guidance = self.cfg_scale_sat * (sat_cond_pred - image_cond_pred)
            return uncond_pred + image_guidance + satellite_guidance
        else:
            # Fallback to original single CFG logic from SevaPipeline
            return super().get_output_cfg(predicted_output, cfg_scale)
