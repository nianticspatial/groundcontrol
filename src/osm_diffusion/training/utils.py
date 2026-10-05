from typing import Any

import torch


def update_checkpoint_with_norm3_to_norm4(state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}
    for k, val in state_dict.items():
        new_state_dict[k] = val
        if "transformer_blocks" in k and "norm3" in k:
            new_k = k.replace("norm3", "norm4")
            if new_k not in state_dict:
                new_state_dict[new_k] = val.clone()

    return new_state_dict


def get_debug_attn_logs(all_debug_dicts, current_dataset_name):
    interesting_keys = ["transformer_blocks_0/attn3/attn", "transformer_blocks_0/attn3/out"]
    interesting_blocks = ["input_ds2", "middle_ds8"]

    key_vals_to_log = {}

    for t in range(len(all_debug_dicts)):
        for block in interesting_blocks:
            for key in interesting_keys:
                if block + "/" + key in all_debug_dicts[t]:
                    val = all_debug_dicts[t][block + "/" + key]
                    key_vals_to_log[f"debug/{current_dataset_name}/{t}__{block}__{key}"] = (
                        torch.norm(val, p=2, dim=list(range(1, val.ndim))).mean()
                    )

    return key_vals_to_log
