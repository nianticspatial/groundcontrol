import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

# --- Helper Functions ---


def make_betas(
    num_timesteps: int, linear_start: float = 1e-4, linear_end: float = 2e-2
) -> np.ndarray:
    """Creates a linear beta schedule."""
    betas = (
        torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# --- Main Scheduler Class ---
class DDPMTrainingScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler that is mathematically consistent with the SEVA inference sampler.

    - The `add_noise` method is used for training.
    - The `step` method is used for inference, mimicking the EulerEDMSampler logic.
    """

    order = 1
    _compatibles = []

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        linear_start: float = 5e-6,
        linear_end: float = 0.012,
        log_snr_shift: float = 2.4,
        prediction_type: str = "epsilon",
        shift_alphas_zero_terminal_snr: bool = False,
        device="cpu",
    ):
        if prediction_type != "epsilon":
            raise ValueError("This scheduler is designed for epsilon-predicting models only.")

        # Store config
        self.num_train_timesteps = num_train_timesteps

        # Calculate alphas_cumprod for sigma generation
        betas = make_betas(num_train_timesteps, linear_start, linear_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        if shift_alphas_zero_terminal_snr:
            self.alpha_shift_zero_terminal_snr()

        # Create the sigmas used for training in the `add_noise` function
        self.train_sigmas = self.get_sigmas(num_train_timesteps)

        # Inference sigmas will be set by `set_timesteps`
        self.sampling_sigmas = None
        self.num_inference_steps = None
        self.timesteps = None  # Initialize timesteps attribute
        self._step_index = None

    def get_sigmas(self, n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a sigma schedule derived from DDPM alphas, consistent with SEVA's DDPMDiscretization.
        """
        if n <= 0:
            return torch.empty(0, device=self.config.device)

        if n < self.num_train_timesteps:
            # Subsample timesteps for inference
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, n, endpoint=True).astype(int)
        elif n == self.num_train_timesteps:
            timesteps = np.arange(self.num_train_timesteps)
        else:
            raise ValueError(
                f"Expected n <= training timesteps {self.num_train_timesteps}, but got n = {n}."
            )

        alphas_cumprod = self.alphas_cumprod[timesteps]
        sigmas = ((1 - alphas_cumprod) / (alphas_cumprod + 1e-8)) ** 0.5

        if self.config.log_snr_shift is not None:
            sigmas = sigmas * np.exp(self.config.log_snr_shift)

        # CORRECTED: Inference requires sigmas in descending order (from max noise to min)
        # The SEVA sampler expects this.
        return torch.from_numpy(sigmas.copy()).to(dtype=dtype, device=self.config.device)

    def alpha_shift_zero_terminal_snr(self):
        alphas_bar_sqrt = np.sqrt(self.alphas_cumprod)
        alphas_bar_sqrt_0 = np.copy(alphas_bar_sqrt[0])
        alphas_bar_sqrt_T = np.copy(alphas_bar_sqrt[-1])
        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        self.alphas_cumprod = alphas_bar_sqrt**2  # Revert sqrt

    @property
    def init_noise_sigma(self) -> torch.Tensor:
        assert self.sampling_sigmas is not None
        # The first sigma in the schedule is the maximum one.
        sigma_max = self.sampling_sigmas[0]
        return (sigma_max**2 + 1.0) ** 0.5

    def set_timesteps(self, num_inference_steps: int, dtype: torch.dtype = torch.float32):
        """
        Sets the sigmas for the inference process.
        """
        self.num_inference_steps = num_inference_steps
        self.sampling_sigmas = self.get_sigmas(num_inference_steps, dtype)

        # Append a final sigma of 0 for the last step, only used if using the step() function.
        self.sampling_sigmas = torch.cat(
            [self.sampling_sigmas, torch.zeros(1).to(self.sampling_sigmas)]
        )

        # `timesteps` in diffusers are often just loop indices
        self.timesteps = torch.arange(0, num_inference_steps, device=self.config.device).flip(0)
        self._step_index = 0

    def sigma_to_training_idx(self, sigma: torch.Tensor):
        dists = sigma - self.train_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def training_idx_to_sigma(self, idx: int | torch.Tensor):
        return self.train_sigmas[idx]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Adds noise to the original samples for training.
        """
        sigmas = self.train_sigmas[timesteps]
        return original_samples + noise * append_dims(sigmas, original_samples.ndim).to(
            original_samples
        )

    def step(
        self,
        model_output: torch.Tensor,  # epsilon_pred
        timestep: int,  # Corresponds to the loop index `i` during inference
        sample: torch.Tensor,  # prev_x_t
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        """
        Performs a single reverse diffusion step using the Euler EDM method.
        """
        if self.sampling_sigmas is None:
            raise RuntimeError("Scheduler `set_timesteps` must be called before `step`.")
        if self._step_index is None:
            self._step_index = timestep

        # 1. Get current and next sigma
        sigma = self.sampling_sigmas[self._step_index]
        next_sigma = self.sampling_sigmas[self._step_index + 1]

        # 2. Apply stochastic churn
        gamma = (
            min(s_churn / self.num_inference_steps, 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        sigma_hat = sigma * (gamma + 1.0)

        if gamma > 0:
            eps = (
                randn_tensor(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype,
                )
                * self.s_noise
            )
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 3. CORRECTED: Convert model_output (epsilon) to predicted original sample (x_0)
        # This math matches the `DiscreteDenoiser`'s scaling logic (`sigma_data` = 1.0)
        c_skip = 1.0 / (sigma_hat**2 + 1.0)
        c_out = -sigma_hat

        # Denoised = c_skip * x_t + c_out * network_output
        pred_original_sample = model_output

        # 4. Calculate ODE derivative `d`
        derivative = (sample - pred_original_sample) / sigma_hat

        # 5. Perform the Euler step
        dt = next_sigma - sigma_hat
        prev_sample = sample + derivative * dt

        # Update step index
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.num_train_timesteps
