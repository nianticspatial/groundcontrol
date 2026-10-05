"""
Generate rel_image_paths.txt file by running `find . -type f -name "*.jpg" -print > rel_image_paths.txt`

$> cd /mnt/nas/personal/daniyar/projects/osm-diffusion/toy_dataset
$> find . -type f -name "*.jpg" -print > re_image_paths.txt
$> cd ~/osm-diffusion
$> CUDA_VISIBLE_DEVICES=0 python -m src.semeval.dinov1 \
    --dataset_path_gt /mnt/nas/personal/daniyar/projects/osm-diffusion/toy_dataset \
    --dataset_path_pred /mnt/nas/personal/daniyar/projects/osm-diffusion/toy_dataset_pred/ \
    --rel_image_paths_txt toy_dataset/rel_image_paths.txt \
    --batch_size 1
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import PIL
import torch
from torch import nn
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from tqdm import tqdm

from osm_diffusion.semeval import dino_utils as vits

dino_normalize = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

dino_transform = pth_transforms.Compose(
    [
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        dino_normalize,
    ]
)

square_transform = pth_transforms.Compose(
    [
        pth_transforms.Resize(224, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        dino_normalize,
    ]
)


class DINOv1SimilarityMeasure(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
        self,
        dino_model_weights_path: Path = Path(
            "/mnt/nas/shared/projects/sem_eval/dino_vitbase16_pretrain.pth"
        ),
        dino_model_type: str = "vit_base",
        dino_patch_size: int = 16,
        reduction: Literal["pixelwise_mean", "mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.reduction = reduction

        assert "vit" in dino_model_type
        self.model = vits.__dict__[dino_model_type](patch_size=dino_patch_size, num_classes=0)
        self.model.cuda()
        load_pretrained_weights(
            self.model, dino_model_weights_path, None, dino_model_type, dino_patch_size
        )
        self.model.eval()

    def extract_features(self, images):
        with torch.no_grad():
            batch_size = images.shape[0]
            images_batch = []
            for i in range(batch_size):
                images_batch.append(
                    square_transform(
                        PIL.Image.fromarray(
                            np.array(
                                255 * images[i].detach().cpu().permute(1, 2, 0).float().numpy(),
                                dtype=np.uint8,
                            )
                        )
                    )
                )
            images_batch = torch.stack(images_batch).to(device=images.device, dtype=images.dtype)
            feats = self.model.get_intermediate_layers(images_batch, n=1)[0]
            feats = feats[:, 1:, :]  # we discard the [CLS] token
            h, w = int(images_batch.shape[2] / self.model.patch_embed.patch_size), int(
                images_batch.shape[3] / self.model.patch_embed.patch_size
            )
            dim = feats.shape[-1]
            feats.reshape(batch_size, h, w, dim)
            return feats

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        with torch.no_grad():
            self.preds.append(self.extract_features(preds.cuda()).detach().cpu())
            self.target.append(self.extract_features(target.cuda()).detach().cpu())

    def compute(self) -> torch.Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        diff = (target - preds).square()
        if self.reduction == "pixelwise_mean":
            return diff.mean(1)
        elif self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        elif self.reduction == "none" or self.reduction is None:
            return diff
        else:
            raise NotImplementedError()


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg)
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, rel_image_paths: list[str], transform):
        self.dataset_path = dataset_path
        self.rel_image_paths = rel_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.rel_image_paths)

    def __getitem__(self, idx):
        image_path = self.dataset_path / self.rel_image_paths[idx]
        image = PIL.Image.open(str(image_path))
        image = self.transform(image)
        return image, self.rel_image_paths[idx]


def dinov1_semeval_pipeline(
    dataset_path_gt: Path,
    dataset_path_pred: Path,
    rel_image_paths_gt: list[str],
    rel_image_paths_pred: list[str],
    batch_size: int,
    num_workers: int,
    score_save_path: Path,
    dino_model_weights_path: Path,
    dino_model_type: str,
    dino_patch_size: int,
    ignore_cache: bool = True,
    save_cache: bool = False,
):
    if score_save_path.is_dir():
        score_save_path = score_save_path / f"scores_dino_{time.time()}.json"
    score_save_path.parent.mkdir(parents=True, exist_ok=True)

    # ============ preparing data ... ============
    dataset_gt = ImagesDataset(dataset_path_gt, rel_image_paths_gt, transform=square_transform)
    feats_gt_pickle_path = (
        dataset_path_gt / f"feats_gt_dinov1_{dino_model_type}_{dino_patch_size}.pickle"
    )
    dataset_pred = ImagesDataset(
        dataset_path_pred, rel_image_paths_pred, transform=square_transform
    )
    feats_pred_pickle_path = (
        dataset_path_pred / f"feats_pred_dinov1_{dino_model_type}_{dino_patch_size}.pickle"
    )
    data_loader_gt = torch.utils.data.DataLoader(
        dataset_gt,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    data_loader_pred = torch.utils.data.DataLoader(
        dataset_pred,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_gt)} gt and {len(dataset_pred)} pred imgs.")

    # ============ building network ... ============
    assert "vit" in dino_model_type
    model = vits.__dict__[dino_model_type](patch_size=dino_patch_size, num_classes=0)
    model.cuda()
    load_pretrained_weights(model, dino_model_weights_path, None, dino_model_type, dino_patch_size)
    model.eval()

    if feats_gt_pickle_path.exists() and not ignore_cache:
        with open(feats_gt_pickle_path, "rb") as f:
            feats_gt = pickle.load(f)
    else:
        print("Extracting features for GT")
        feats_gt = extract_features(model, data_loader_gt, use_cuda=True)
        if save_cache:
            with open(feats_gt_pickle_path, "wb") as f:
                pickle.dump(feats_gt, f)

    if feats_pred_pickle_path.exists() and not ignore_cache:
        with open(feats_pred_pickle_path, "rb") as f:
            feats_pred = pickle.load(f)
    else:
        print("Extracting features for pred")
        feats_pred = extract_features(model, data_loader_pred, use_cuda=True)
        if save_cache:
            with open(feats_pred_pickle_path, "wb") as f:
                pickle.dump(feats_pred, f)

    scores = {"mean_sqe": {}, "mean_abs": {}, "mean_cos": {}}
    for k in feats_gt.keys():
        scores["mean_sqe"][k] = (
            (feats_gt[k] - feats_pred[k.replace("_gt", "_gen")]).square().mean().item()
        )
        scores["mean_abs"][k] = (
            (feats_gt[k] - feats_pred[k.replace("_gt", "_gen")]).abs().mean().item()
        )
        scores["mean_cos"][k] = (
            (
                (feats_gt[k] * feats_pred[k.replace("_gt", "_gen")]).sum(1)
                / (
                    1e-4
                    + torch.sqrt(feats_gt[k].square().sum(1))
                    * torch.sqrt(feats_pred[k.replace("_gt", "_gen")].square().sum(1))
                )
            )
            .mean()
            .item()
        )

    mse = torch.tensor(list(scores["mean_sqe"].values())).mean()
    print(f"Mean square error: {mse}")
    print(f"Mean absolute difference: {torch.tensor(list(scores['mean_abs'].values())).mean()}")
    print(f"Mean cosine similarity: {torch.tensor(list(scores['mean_cos'].values())).mean()}")
    with open(score_save_path, "w") as f:
        json.dump(scores, f)
    return mse


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    features = {}
    for images, batch_rel_image_paths in tqdm(data_loader):
        if use_cuda:
            images = images.cuda(non_blocking=True)
        # feats = model(images).clone()
        batch_size = images.shape[0]
        feats = model.get_intermediate_layers(images, n=1)[0]
        feats = feats[:, 1:, :]  # we discard the [CLS] token
        h, w = int(images.shape[2] / model.patch_embed.patch_size), int(
            images.shape[3] / model.patch_embed.patch_size
        )
        dim = feats.shape[-1]
        feats = feats.reshape(batch_size, h, w, dim)

        for i, rel_image_path in enumerate(batch_rel_image_paths):
            features[rel_image_path] = feats[i].cpu()
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic similarity using DINO v1 encoder."
    )
    parser.add_argument(
        "--dataset_path_gt",
        type=Path,
        required=True,
        help="Root folder of the dataset with ground truths.",
    )
    parser.add_argument(
        "--dataset_path_pred",
        type=Path,
        required=True,
        help="Root folder of the dataset with predictions.",
    )
    parser.add_argument(
        "--rel_image_paths_txt",
        type=Path,
        required=True,
        help="Txt file that contains relative paths to images. The relative paths are shared between dataset_path_gt and dataset_path_pred.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for Dataloaders",
    )
    parser.add_argument(
        "--score_save_path", type=Path, default=Path(""), help="Save scores as JSON at this path"
    )
    parser.add_argument(
        "--dino_model_weights_path",
        type=Path,
        default=Path("/mnt/nas/shared/projects/sem_eval/dino_vitbase16_pretrain.pth"),
        help="Path to the DINO model weights (.pth) file.",
    )
    parser.add_argument(
        "--dino_model_type",
        type=str,
        default="vit_base",
        help="DINO model type (vit_base, vit_small, vit_tiny)",
    )
    parser.add_argument(
        "--dino_patch_size",
        type=int,
        default=16,
        help="DINO model patch size (16 or 8)",
    )
    args = parser.parse_args()

    with open(args.rel_image_paths_txt, "r") as f:
        rel_image_paths_gt = [p.strip("\n") for p in f.readlines()]
    rel_image_paths_pred = [p.replace("_gt", "_gen") for p in rel_image_paths_gt]

    dinov1_semeval_pipeline(
        dataset_path_gt=args.dataset_path_gt,
        dataset_path_pred=args.dataset_path_pred,
        rel_image_paths_gt=rel_image_paths_gt,
        rel_image_paths_pred=rel_image_paths_pred,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        score_save_path=args.score_save_path,
        dino_model_weights_path=args.dino_model_weights_path,
        dino_model_type=args.dino_model_type,
        dino_patch_size=args.dino_patch_size,
    )


if __name__ == "__main__":
    main()
