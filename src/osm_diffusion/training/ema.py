import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


class EMAUnetAveraging(pl.Callback):
    """
    Maintains an Exponential Moving Average (EMA) of the trainable UNet (ldm) weights only.

    - Averages ONLY `pipeline.ldm` parameters (does not touch the VAE).
    - Can optionally average only trainable parameters (requires_grad=True) via `trainable_only`.
    - Update frequency can be throttled via `every_n_steps`.
    - EMA can start after `start_after_steps`.
    - EMA weights can be kept on a specified device (cpu/cuda/auto).
    - During validation, swaps in EMA weights; restores original after.
    - Persists EMA state across checkpoints.
    """

    def __init__(
        self,
        enabled: bool = False,
        device: str = "cpu",
        every_n_steps: int = 1,
        start_after_steps: int = 0,
        decay: float = 0.999,
        trainable_only: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.device_str = str(device)
        self.every_n_steps = int(max(1, every_n_steps))
        self.start_after_steps = int(max(0, start_after_steps))
        self.decay = float(decay)
        self.trainable_only = bool(trainable_only)

        # For full model EMA
        self._ema_model: AveragedModel | None = None
        # For trainable-only EMA (memory efficient)
        self._ema_params: dict[str, torch.Tensor] | None = None

        self._is_swapped: bool = False
        self._pending_state_dict: dict | None = None

    def _resolve_device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_str == "cuda":
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_unet(self, pipeline: nn.Module) -> nn.Module:
        return pipeline.ldm

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None
    ) -> None:
        if not self.enabled:
            return

        device = self._resolve_device()
        unet = self._get_unet(pl_module.pipeline)

        if self.trainable_only:
            # Memory-efficient mode: only store trainable parameters
            self._ema_params = {}
            trainable_count = 0
            total_params = 0

            for name, param in unet.named_parameters():
                if param.requires_grad:
                    # Clone and move to target device
                    self._ema_params[name] = param.detach().clone().to(device)
                    trainable_count += 1
                    total_params += param.numel()

            print(
                f"[EMA] Memory-efficient mode: Tracking {trainable_count} trainable parameters "
                f"({total_params:,} total elements, ~{total_params * 4 / 1024**2:.1f} MB)"
            )

            # Load pending state if resuming
            if self._pending_state_dict is not None:
                print("[EMA] Loading EMA pending state dict")
                for name, param in self._pending_state_dict.items():
                    if name in self._ema_params:
                        self._ema_params[name].data.copy_(param.to(device))
                self._pending_state_dict = None
        else:
            # Full model EMA mode
            avg_fn = get_ema_avg_fn(self.decay)
            self._ema_model = AveragedModel(unet, device=device, avg_fn=avg_fn)
            print(f"[EMA] Full model mode: Tracking all parameters")

            # Load pending state if resuming
            if self._pending_state_dict is not None:
                print("[EMA] Loading EMA pending state dict into model")
                self._ema_model.load_state_dict(self._pending_state_dict)
                self._pending_state_dict = None

    # --- Training updates (run once per optimizer step, respecting grad accumulation) ---
    def on_before_zero_grad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        if not self.enabled:
            return
        if self._ema_model is None and self._ema_params is None:
            return

        global_step = int(trainer.global_step)
        if global_step < self.start_after_steps:
            return
        if (global_step % self.every_n_steps) != 0:
            return

        unet = self._get_unet(pl_module.pipeline)

        if self.trainable_only:
            self._update_trainable_parameters(unet)
        else:
            self._ema_model.update_parameters(unet)

    def _update_trainable_parameters(self, model: nn.Module) -> None:
        """Update only the trainable parameters in the EMA (memory-efficient mode)."""
        if self._ema_params is None:
            return

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, ema_param in self._ema_params.items():
                if name in model_params:
                    param = model_params[name]
                    # Apply EMA update: ema = decay * ema + (1 - decay) * current
                    ema_param.mul_(self.decay).add_(
                        param.to(ema_param.device), alpha=1 - self.decay
                    )

    # --- Swap in EMA weights for validation ---
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        if self._ema_model is None and self._ema_params is None:
            return
        # Swap parameters and buffers between EMA model and current UNet
        print("[EMA] Swapping EMA weights into model for validation")
        self._swap_with_ema_params(pl_module)
        self._is_swapped = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        if self._ema_model is None and self._ema_params is None:
            return
        if not self._is_swapped:
            return
        # Swap back to restore original UNet params
        print("[EMA] Restoring original UNet params after validation")
        self._swap_with_ema_params(pl_module)
        self._is_swapped = False

    def _swap_with_ema_params(self, pl_module: pl.LightningModule) -> None:
        unet = pl_module.pipeline.ldm

        with torch.no_grad():
            if self.trainable_only and self._ema_params is not None:
                # Memory-efficient mode: swap only trainable parameters from dict
                unet_params = dict(unet.named_parameters())
                for name, ema_param in self._ema_params.items():
                    if name in unet_params:
                        param = unet_params[name]
                        # Swap
                        tmp = ema_param.data.clone()
                        ema_param.data.copy_(param.data.to(ema_param.device))
                        param.data.copy_(tmp.to(param.device))

            elif self._ema_model is not None:
                # Full model mode: swap all parameters and buffers
                ema_module = self._ema_model.module

                # Swap parameters
                ema_params_dict = dict(ema_module.named_parameters())
                for n_cur, p_cur in unet.named_parameters():
                    p_avg = ema_params_dict[n_cur]
                    tmp = p_avg.data.clone()
                    p_avg.data.copy_(p_cur.data.to(p_avg.device))
                    p_cur.data.copy_(tmp.to(p_cur.device))

                # Swap buffers
                # Note: Buffers are non-trainable tensors like BatchNorm running stats, positional encodings, etc.
                ema_buffers = dict(ema_module.named_buffers())
                for name, buf_cur in unet.named_buffers():
                    if name in ema_buffers:
                        buf_avg = ema_buffers[name]
                        tmp = buf_avg.data.clone()
                        buf_avg.data.copy_(buf_cur.data.to(buf_avg.device))
                        buf_cur.data.copy_(tmp.to(buf_cur.device))

    # --- Persist and restore EMA state across checkpoints ---
    def state_dict(self) -> dict:
        if not self.enabled:
            return {}

        if self.trainable_only and self._ema_params is not None:
            # Memory-efficient mode: save parameter dict directly
            return {
                "ema_params": {name: param.cpu() for name, param in self._ema_params.items()},
                "mode": "trainable_only",
            }
        elif self._ema_model is not None:
            # Full model mode: save AveragedModel state
            return {"ema_state": self._ema_model.state_dict(), "mode": "full_model"}
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        if not self.enabled:
            return

        mode = state_dict.get("mode", "full_model")  # Default to old format

        if mode == "trainable_only" and "ema_params" in state_dict:
            # Loading memory-efficient mode checkpoint
            ema_params = state_dict["ema_params"]
            print(f"[EMA] Loading {len(ema_params)} trainable parameters from checkpoint")

            # If not yet initialized, stash and load later in setup
            if self._ema_params is None:
                print("[EMA] Stashing EMA params for later loading")
                self._pending_state_dict = ema_params
            else:
                # Load directly
                device = self._resolve_device()
                for name, param in ema_params.items():
                    if name in self._ema_params:
                        self._ema_params[name].data.copy_(param.to(device))

        elif "ema_state" in state_dict:
            # Loading full model mode checkpoint
            ema_state = state_dict["ema_state"]
            print("[EMA] Loading full model EMA state from checkpoint")

            # If not yet initialized, stash and load later in setup
            if self._ema_model is None:
                print("[EMA] Stashing EMA state dict for later loading")
                self._pending_state_dict = ema_state
            else:
                self._ema_model.load_state_dict(ema_state)
        else:
            print("[EMA] WARNING: EMA state not found in checkpoint")
