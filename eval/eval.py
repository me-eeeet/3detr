"""Eval Predications."""

import sys
sys.path.append("../")

from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm

from datasets.kitti_copy import KITTIDetectionDataset, KITTIDatasetConfig
from utils.eval_det import eval_det_multiprocessing


def metrics_to_str(ap_iou_thresholds, overall_ret, per_class=True):
    mAP_strs = []
    AR_strs = []
    per_class_metrics = []
    for ap_iou_thresh in ap_iou_thresholds:
        mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
        mAP_strs.append(f"{mAP:.2f}")
        ar = overall_ret[ap_iou_thresh]["AR"] * 100
        AR_strs.append(f"{ar:.2f}")

        if per_class:
            # per-class metrics
            per_class_metrics.append("-" * 5)
            per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
            for x in list(overall_ret[ap_iou_thresh].keys()):
                if x == "mAP" or x == "AR":
                    pass
                else:
                    met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                    per_class_metrics.append(met_str)

    ap_header = [f"mAP{x:.2f}" for x in ap_iou_thresholds]
    ap_str = ", ".join(ap_header)
    ap_str += ": " + ", ".join(mAP_strs)
    ap_str += "\n"

    ar_header = [f"AR{x:.2f}" for x in ap_iou_thresholds]
    ap_str += ", ".join(ar_header)
    ap_str += ": " + ", ".join(AR_strs)

    if per_class:
        per_class_metrics = "\n".join(per_class_metrics)
        ap_str += "\n"
        ap_str += per_class_metrics

    return ap_str


def eval(
    dataset_config: KITTIDatasetConfig,
    dataset: KITTIDetectionDataset,
    predictions_dir: Path,
) -> None:
    
    all_ground_truths = {}
    all_predictions = {}

    for index in tqdm(range(len(dataset))):
        data = dataset[index]
        ground_truth_corners = data["gt_box_corners"]
        ground_truth_labels = data["gt_box_sem_cls_label"]
        ground_truth_present = data["gt_box_present"]
        # ground_truth_difficulty = data["gt_difficulty"]

        ground_truths = [
            (ground_truth_labels[j], ground_truth_corners[j])
            for j in range(ground_truth_corners.shape[0])
            if ground_truth_present[j] == 1
        ]
        all_ground_truths[index] = ground_truths

        predictions: dict[str, np.ndarray] = np.load(predictions_dir / f"{index}.npy", allow_pickle=True).item()
        all_predictions[index] = list(zip(*predictions.values()))
    
    # compute metrics
    ap_iou_thresholds = [0.25, 0.5, 0.7]

    overall_ret = OrderedDict()
    for ap_iou_thresh in ap_iou_thresholds:
        ret_dict = OrderedDict()
        rec, prec, ap = eval_det_multiprocessing(
            all_predictions, all_ground_truths, ovthresh=ap_iou_thresh
        )
        
        for key in sorted(ap.keys()):
            clsname = dataset_config.class2type[key] if dataset_config.class2type else str(key)
            ret_dict["%s Average Precision" % (clsname)] = ap[key]
        
        ap_vals = np.array(list(ap.values()), dtype=np.float32)
        ap_vals[np.isnan(ap_vals)] = 0
        ret_dict["mAP"] = ap_vals.mean()
        
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = dataset_config.class2type[key] if dataset_config.class2type else str(key)
            try:
                ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict["%s Recall" % (clsname)] = 0
                rec_list.append(0)
        
        ret_dict["AR"] = np.mean(rec_list)
        overall_ret[ap_iou_thresh] = ret_dict
    
    return metrics_to_str(ap_iou_thresholds, overall_ret)


if __name__ == "__main__":

    parser = ArgumentParser(description="KITTI Evaluator")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        help="KITTI data root directory path",
        default="/common/dataset/kitti/object"
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        help="Data split type",
        choices=["train", "val", "trainval", "test"],
        default="train",
    )
    parser.add_argument(
        "-p",
        "--predictions_dir",
        type=Path,
        help="Predictions Directory",
        required=True,
    )

    args = parser.parse_args()

    config = KITTIDatasetConfig()
    dataset = KITTIDetectionDataset(
        dataset_config=config,
        split_set=args.split,
        root_dir=args.root_dir,
        augment=False,
    )

    metrics = eval(
        dataset_config=config,
        dataset=dataset,
        predictions_dir=args.predictions_dir,
    )
    print(metrics)