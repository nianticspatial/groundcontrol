"""
Segment Anything similarity metric for semantic evaluation.

Pass sam_model_weights_path, or set sam_weights in paths.yaml / GC_SAM_WEIGHTS.
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
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from tqdm import tqdm

# moving this inside the functions to avoid package issues when importing Measures while they are not used
# from segment_anything import SamPredictor, sam_model_registry


class SegAnySimilarityMeasure(Metric):
    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
        self,
        sam_model_weights_path: Path | str | None = None,
        sam_model_type: str = "default",
        reduction: Literal["pixelwise_mean", "mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        from segment_anything import SamPredictor, sam_model_registry

        super().__init__(**kwargs)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.reduction = reduction

        weights_path = sam_model_weights_path or os.environ.get("GC_SAM_WEIGHTS")
        if not weights_path:
            raise ValueError(
                "SAM weights path required: set GC_SAM_WEIGHTS or pass sam_model_weights_path."
            )
        sam = sam_model_registry[sam_model_type](checkpoint=str(weights_path))
        sam.cuda()
        sam.eval()
        self.predictor = SamPredictor(sam)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        for i in range(preds.shape[0]):
            preds_image = np.array(
                255 * preds[i].detach().cpu().permute(1, 2, 0).float().numpy(), dtype=np.uint8
            )
            target_image = np.array(
                255 * target[i].detach().cpu().permute(1, 2, 0).float().numpy(), dtype=np.uint8
            )
            self.predictor.set_image(preds_image)
            self.preds.append(self.predictor.features.detach().clone().cpu())
            self.predictor.set_image(target_image)
            self.target.append(self.predictor.features.detach().clone().cpu())

    def compute(self) -> torch.Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        # cosine similarity
        diff = (target * preds).sum(1) / (
            1e-4 + torch.sqrt(target.square().sum(1)) * torch.sqrt(preds.square().sum(1))
        )

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


@torch.no_grad()
def extract_features(predictor, dataset_folder, rel_image_paths):
    feats = {}
    for rel_image_path in tqdm(rel_image_paths):
        image = PIL.Image.open(dataset_folder / rel_image_path)
        predictor.set_image(np.array(image))
        features = predictor.features
        feats[str(rel_image_path)] = features.cpu().clone()
    return feats


def extract_feature_pipeline(
    dataset_path_gt: Path,
    dataset_path_pred: Path,
    rel_image_paths: list[str],
    score_save_path: Path,
    sam_model_weights_path: Path,
    sam_model_type: str = "default",
):
    from segment_anything import SamPredictor, sam_model_registry

    if score_save_path.is_dir():
        score_save_path = score_save_path / f"scores_sam_{time.time()}.json"
    score_save_path.parent.mkdir(parents=True, exist_ok=True)

    sam = None
    predictor = None

    feats_gt_pickle_path = dataset_path_gt / f"feats_gt_sam_{sam_model_type}.pickle"
    feats_pred_pickle_path = dataset_path_pred / f"feats_pred_sam_{sam_model_type}.pickle"

    if feats_gt_pickle_path.exists():
        with open(feats_gt_pickle_path, "rb") as f:
            feats_gt = pickle.load(f)
    else:
        print("Extracting features for gt")
        if sam is None:
            sam = sam_model_registry[sam_model_type](checkpoint=sam_model_weights_path)
            sam.cuda()
            sam.eval()
            predictor = SamPredictor(sam)

        feats_gt = extract_features(predictor, dataset_path_gt, rel_image_paths)
        with open(feats_gt_pickle_path, "wb") as f:
            pickle.dump(feats_gt, f)

    if feats_pred_pickle_path.exists():
        with open(feats_pred_pickle_path, "rb") as f:
            feats_pred = pickle.load(f)
    else:
        print("Extracting features for pred")
        if sam is None:
            sam = sam_model_registry[sam_model_type](checkpoint=sam_model_weights_path)
            sam.cuda()
            sam.eval()
            predictor = SamPredictor(sam)
        rel_image_paths_pred = [p.replace("_gt", "_gen") for p in rel_image_paths]
        feats_pred = extract_features(predictor, dataset_path_pred, rel_image_paths_pred)
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

    print(f"Mean square error: {torch.tensor(list(scores['mean_sqe'].values())).mean()}")
    print(f"Mean absolute difference: {torch.tensor(list(scores['mean_abs'].values())).mean()}")
    print(f"Mean cosine similarity: {torch.tensor(list(scores['mean_cos'].values())).mean()}")
    with open(score_save_path, "w") as f:
        json.dump(scores, f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity using SAM.")
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
        "--score_save_path", type=Path, default=Path(""), help="Save scores as JSON at this path"
    )
    parser.add_argument(
        "--sam_model_weights_path",
        type=Path,
        default=None,
        help="Path to the SAM model weights (.pth) file. Defaults to GC_SAM_WEIGHTS.",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="default",  # vit_h
        help="SAM model type (default i.e. vit_h, vit_b, vit_s)",
    )
    args = parser.parse_args()

    with open(args.rel_image_paths_txt, "r") as f:
        rel_image_paths = [p.strip("\n") for p in f.readlines()]

    extract_feature_pipeline(
        args.dataset_path_gt,
        args.dataset_path_pred,
        rel_image_paths,
        score_save_path=args.score_save_path,
        sam_model_weights_path=args.sam_model_weights_path,
        sam_model_type=args.sam_model_type,
    )


if __name__ == "__main__":
    main()
