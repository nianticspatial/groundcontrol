"""Evaluation-time perturbations for satellite conditioning."""

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SatellitePerturbationConfig:
    rotate: bool = False
    random_other: bool = False
    flip: bool = False
    zero: bool = False
    seed: int = 42

    @property
    def enabled(self) -> bool:
        return self.rotate or self.random_other or self.flip or self.zero


class SatellitePerturbationDataset(Dataset):
    """Wrap a dataset and perturb its satellite conditioning deterministically."""

    def __init__(
        self,
        dataset: Dataset,
        config: SatellitePerturbationConfig,
        use_precomputed_features: bool,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.conditioning_key = (
            "satellite_features" if use_precomputed_features else "satellite_images"
        )
        if config.random_other and len(dataset) < 2:
            raise ValueError(
                "--random_satellite requires an evaluation dataset with at least two examples."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def _rng(self, index: int) -> random.Random:
        payload = f"{self.config.seed}|{index}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(seed)

    @staticmethod
    def _spatial_dims(key: str) -> tuple[int, int]:
        # Cached features are [..., H, W, C], while images are [..., C, H, W].
        return (-3, -2) if key == "satellite_features" else (-2, -1)

    @staticmethod
    def _get_tensor(sample: dict[str, Any], key: str, index: int) -> torch.Tensor:
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Example {index} does not contain tensor-valued {key!r}; "
                "satellite perturbation cannot be applied."
            )
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = copy.copy(self.dataset[index])
        rng = self._rng(index)
        donor = None
        rotation_k = rng.choice((1, 3)) if self.config.rotate else None

        if self.config.random_other:
            donor_index = rng.randrange(len(self.dataset) - 1)
            if donor_index >= index:
                donor_index += 1
            donor = self.dataset[donor_index]

        # Keep the raw image synchronized for debug visualizations when cached
        # features are the actual model input.
        keys = [self.conditioning_key]
        if self.conditioning_key == "satellite_features" and "satellite_images" in sample:
            keys.append("satellite_images")

        for key in keys:
            value = self._get_tensor(sample, key, index)
            if donor is not None:
                donor_value = self._get_tensor(donor, key, donor_index)
                if donor_value.shape != value.shape:
                    raise ValueError(
                        f"Cannot replace {key} for example {index}: source shape "
                        f"{tuple(value.shape)} differs from donor {donor_index} shape "
                        f"{tuple(donor_value.shape)}."
                    )
                value = donor_value.clone()
            else:
                value = value.clone()

            spatial_dims = self._spatial_dims(key)
            if rotation_k is not None:
                value = torch.rot90(value, k=rotation_k, dims=spatial_dims)
            if self.config.flip:
                value = torch.flip(value, dims=(spatial_dims[1],))
            if self.config.zero:
                value = torch.zeros_like(value)
            sample[key] = value

        return sample
