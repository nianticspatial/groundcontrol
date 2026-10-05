import torch

import osm_diffusion.seva.utils as seva_utils
from osm_diffusion.satellite.satellite_ray_fused_query_ca_pipeline import (
    SatelliteRayFusedQueryCAPipeline,
)
from osm_diffusion.seva.model import SevaParams
from osm_diffusion.seva.modules.autoencoder import AutoEncoder
from osm_diffusion.seva.modules.conditioner import CLIPConditioner
from osm_diffusion.seva.satellite_model import SevaSatRayAttnFusedQuerySeparateBlock
from osm_diffusion.training.ddpm_scheduler import DDPMTrainingScheduler

MODEL_NAME = "satellite_ray_fused_query_ca_sep_blocks"


def get_satellite_pipeline_kwargs(model_config):
    return {
        "per_view_satellite_cond": model_config.get("per_view_satellite_cond", False),
        "disable_sat_conditioning": model_config.get("disable_sat_conditioning", False),
        "train_sat_encoder": model_config.get("train_sat_encoder", False),
        "train_sat_dpt_head": model_config.get("train_sat_dpt_head", False),
        "sat_encoder_name": model_config.get("sat_encoder_name", "dinov2_vits14"),
        "sat_use_dpt_head": model_config.get("sat_use_dpt_head", False),
        "sat_dpt_head_config": model_config.get("sat_dpt_head_config"),
        "drop_clip_conditioning": model_config.get("drop_clip_conditioning", False),
    }


def create_pipeline(
    model_config,
    paths_config,
    dtype=None,
    return_attn=False,
):
    """Build the released satellite-conditioned SEVA pipeline."""
    if model_config.name != MODEL_NAME:
        raise ValueError(
            f"This release only supports model '{MODEL_NAME}', got '{model_config.name}'."
        )

    pretrained_weights_path = paths_config.pretrained_weights_path
    vae_ckpt_path = paths_config.vae_ckpt_path
    clip_hf_hub_path = paths_config.clip_hf_hub_path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Initializing {model_config.name} pipeline...")
    skip_clip_attn = model_config.get("skip_clip_attn", False)
    sat_cond_dim = model_config.get("sat_cond_dim", 384)
    drop_clip_conditioning = model_config.get("drop_clip_conditioning", False)

    vae = AutoEncoder(chunk_size=1, ckpt_path=vae_ckpt_path).to(device).eval()
    if not skip_clip_attn and not drop_clip_conditioning:
        semantic_encoder = CLIPConditioner(hf_hub_path=clip_hf_hub_path).to(device).eval()
    else:
        semantic_encoder = None

    scheduler = DDPMTrainingScheduler(
        device=device,
        shift_alphas_zero_terminal_snr=model_config.get("shift_alphas_zero_terminal_snr", False),
    )
    sat_kwargs = get_satellite_pipeline_kwargs(model_config)
    additional_model_kwargs = {
        "skip_clip_attn": skip_clip_attn,
        "sat_cond_dim": sat_cond_dim,
    }
    model_cls = SevaSatRayAttnFusedQuerySeparateBlock
    pipeline_cls = SatelliteRayFusedQueryCAPipeline
    sat_kwargs.update(
        {
            "sample_points_min_depth": model_config.get("sample_points_min_depth", 2.0),
            "sample_points_max_depth": model_config.get("sample_points_max_depth", 100.0),
            "num_sample_points": model_config.get("num_sample_points", 16),
            "ray_pos_enc_freqs": model_config.get("ray_pos_enc_freqs", 10),
            "cfg_scale_image": model_config.get("cfg_scale_image", 2.0),
            "cfg_scale_sat": model_config.get("cfg_scale_sat", 2.0),
            "use_dual_cfg": model_config.get("use_dual_cfg", False),
            "strict_dual_cfg_sat_mask": model_config.get("strict_dual_cfg_sat_mask", False),
            "add_positional_encoding": not model_config.get("use_mlp_sat_pos", False),
            "position_encoding_scale": model_config.get("position_encoding_scale", 1.0),
            "spherical_sampling": model_config.get("spherical_sampling", False),
            "use_preprocessed_dino_features": model_config.get(
                "use_preprocessed_dino_features", False
            ),
            "reference_translation_noise_m": model_config.get("reference_translation_noise_m", 0.0),
            "reference_rotation_noise_deg": model_config.get("reference_rotation_noise_deg", 0.0),
        }
    )
    additional_model_kwargs["embedding_dim"] = model_config.get("embedding_dim", 256)
    additional_model_kwargs["street_raymap_dim"] = model_config.get("street_raymap_dim", 256)
    additional_model_kwargs["use_mlp_for_sat_ca"] = model_config.get("use_mlp_for_sat_ca", True)
    additional_model_kwargs["use_mlp_for_context"] = model_config.get("use_mlp_for_context", False)
    additional_model_kwargs["sat_ca_before_sa"] = model_config.get("sat_ca_before_sa", False)
    additional_model_kwargs["use_mlp_at_output"] = model_config.get("use_mlp_at_output", False)
    additional_model_kwargs["use_mlp_for_merging_query"] = model_config.get(
        "use_mlp_for_merging_query", False
    )
    additional_model_kwargs["use_mlp_sat_pos"] = model_config.get("use_mlp_sat_pos", False)
    additional_model_kwargs["residual_sat_block"] = model_config.get("residual_sat_block", True)
    additional_model_kwargs["sat_block_use_self_attention"] = model_config.get(
        "sat_block_use_self_attention", False
    )
    additional_model_kwargs["sat_block_in_encoder"] = model_config.get("sat_block_in_encoder", True)
    additional_model_kwargs["sat_block_in_decoder"] = model_config.get(
        "sat_block_in_decoder", False
    )
    additional_model_kwargs["use_rope_sat_attn"] = model_config.get("use_rope_sat_attn", False)
    additional_model_kwargs["use_view_embedding"] = model_config.get("use_view_embedding", True)
    additional_model_kwargs["use_height_cond"] = model_config.get("use_height_cond", False)
    additional_model_kwargs["height_cond_dim"] = model_config.get("height_cond_dim", 328)
    additional_model_kwargs["num_height_cond_bins"] = model_config.get("num_height_cond_bins", 200)
    additional_model_kwargs["use_input_mask"] = model_config.get("use_input_mask", False)
    additional_model_kwargs["satellite_blocks_depth"] = model_config.get(
        "satellite_blocks_depth", 1
    )
    additional_model_kwargs["use_context_norm"] = model_config.get("use_context_norm", False)
    additional_model_kwargs["height_cond_before_sat_cond"] = model_config.get(
        "height_cond_before_sat_cond", False
    )
    additional_model_kwargs["num_satellite_res"] = model_config.get("num_satellite_res", 1)
    additional_model_kwargs["use_sat_output_norm"] = model_config.get("use_sat_output_norm", False)
    additional_model_kwargs["use_sat_output_gate"] = model_config.get("use_sat_output_gate", False)
    additional_model_kwargs["use_mlp_output_gate"] = model_config.get("use_mlp_output_gate", False)
    additional_model_kwargs["use_register_sat_ray_embeddings"] = model_config.get(
        "use_register_sat_ray_embeddings", False
    )
    additional_model_kwargs["scale_shift_predictor"] = model_config.get(
        "scale_shift_predictor", False
    )
    additional_model_kwargs["use_layernorm_after_mlps"] = model_config.get(
        "use_layernorm_after_mlps", False
    )
    additional_model_kwargs["sat_attn_init_to_zero"] = model_config.get(
        "sat_attn_init_to_zero", True
    )
    additional_model_kwargs["sat_attn_num_heads_factor"] = model_config.get(
        "sat_attn_num_heads_factor", 1
    )
    additional_model_kwargs["sat_attn_d_head_factor"] = model_config.get(
        "sat_attn_d_head_factor", 1
    )
    additional_model_kwargs["sat_attn_num_heads"] = model_config.get("sat_attn_num_heads", None)
    additional_model_kwargs["sat_attn_d_head"] = model_config.get("sat_attn_d_head", None)
    additional_model_kwargs["sat_attn_dropout"] = model_config.get("sat_attn_dropout", 0.0)

    print("Using model ", model_cls.__name__)
    params = SevaParams()
    if pretrained_weights_path is None:
        ldm = model_cls(params, **additional_model_kwargs).to(device).to(torch.bfloat16)
    else:
        print("loading model from pretrained weights at ", pretrained_weights_path)
        state_dict = seva_utils.get_model_state_dict(
            model_version=1.1,
            pretrained_model_name_or_path=pretrained_weights_path,
            weight_name="model.safetensors",
            device="cpu",
        )
        ldm = model_cls(params, **additional_model_kwargs).to(torch.bfloat16)
        missing, unexpected = ldm.load_state_dict(state_dict, strict=False)
        ldm.to(device)
        seva_utils.print_load_warning(missing, unexpected)

    pipeline = pipeline_cls(
        ldm=ldm,
        vae=vae,
        semantic_encoder=semantic_encoder,
        scheduler=scheduler,
        device=device,
        cfg_scale=model_config.cfg_scale,
        dtype=dtype,
        use_reference_rotation=model_config.use_reference_rotation,
        using_older_ckpt=model_config.using_older_ckpt,
        return_attn=return_attn,
        img_h=model_config.get("img_h", 256),
        img_w=model_config.get("img_w", 256),
        **sat_kwargs,
    )
    pipeline.to(device)
    return pipeline
