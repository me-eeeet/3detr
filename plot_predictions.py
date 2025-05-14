"""Plot Predictions."""

import sys
sys.path.append("third_party/pointnet2")

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import torch
from tqdm.autonotebook import tqdm

from datasets.kitti import KITTIDatasetConfig, KITTIDetectionDataset
from utils.plotting import get_figure, plot_point_cloud, plot_3d_bbox
from third_party.pointnet2.pointnet2_utils import furthest_point_sample


DIFFICULTY_MAP = {
    0: "easy",
    1: "medium",
    2: "hard",
    -1: "none"
}


def get_parser() -> ArgumentParser:

    parser = ArgumentParser()

    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--predictions_dir", type=str)
    parser.add_argument("--index", type=int)
    parser.add_argument("--radius", default=np.inf, type=float)
    parser.add_argument("--furthest_point_sampling", type=int, default=None)
    parser.add_argument("--save_all", default=False, action="store_true")
    parser.add_argument("--output_dir", type=Path, default="predictions_images")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    dataset_config  = KITTIDatasetConfig()
    dataset = KITTIDetectionDataset(
        dataset_config=dataset_config,
        split_set=args.split,
        root_dir=args.dataset_dir,
        radius=args.radius,
        augment=False,
    )
    print(f"Dataset length: {len(dataset)}")

    predictions_dir = Path(args.predictions_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_all:
        args.index = range(len(dataset))
    else:
        args.index = [args.index]
    
    for index in tqdm(args.index):
        
        # read prediction file
        predictions: dict[str, np.ndarray] = np.load(predictions_dir / f"{index}.npy", allow_pickle=True).item()

        data = dataset.__getitem__(index=index)
        point_cloud = data["point_clouds"]

        if args.furthest_point_sampling:
            batch_point_cloud = torch.tensor(point_cloud[:, :3])[None, ...]
            batch_point_cloud = batch_point_cloud.cuda()
            indexes = furthest_point_sample(batch_point_cloud, args.furthest_point_sampling)
            indexes = indexes.cpu().numpy()[0]
            point_cloud = point_cloud[indexes]
        
        ground_truth_corners = data["gt_box_corners"]
        ground_truth_labels = data["gt_box_sem_cls_label"]
        ground_truth_present = data["gt_box_present"]
        ground_truth_difficulty = data["gt_difficulty"]

        # plot predictions
        fig = get_figure(rows=1, cols=2, titles=["GroundTruth", "Predictions"])
        plot_point_cloud(data=point_cloud, name=dataset.filenames[index], fig=fig, row=1, col=1)
        plot_point_cloud(data=point_cloud, name=dataset.filenames[index], fig=fig, row=1, col=2)
        
        n = len(predictions["ids"])
        for idx in range(n):
            name = dataset_config.class2type[predictions["ids"][idx]]
            points_3d = predictions["corners"][idx]
            score = predictions["scores"][idx]
            plot_3d_bbox(points_3d=points_3d, name=name, fig=fig, extra_text=f"{score:0.2f}", row=1, col=2)
        
        # plot ground truths
        for idx in range(len(ground_truth_corners)):
            if ground_truth_present[idx]:
                name = dataset_config.class2type[ground_truth_labels[idx]]
                difficulty = DIFFICULTY_MAP[ground_truth_difficulty[idx]]
                plot_3d_bbox(points_3d=ground_truth_corners[idx], name=name, fig=fig, extra_text=difficulty, row=1, col=1)
            else:
                break

        if not args.save_all:
            fig.show()
        else:
            fig.write_html(str(args.output_dir / f"{dataset.filenames[index]}.html"))
