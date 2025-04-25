"""Plot Predictions."""

from argparse import ArgumentParser
import numpy as np
from pathlib import Path

from datasets import KITTIDatasetConfig, KITTIDetectionDataset
from utils.plotting import get_figure, plot_point_cloud, plot_3d_bbox


def get_parser() -> ArgumentParser:

    parser = ArgumentParser()

    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--predictions_dir", type=str)
    parser.add_argument("--index", type=int)

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    dataset_config  = KITTIDatasetConfig()
    dataset = KITTIDetectionDataset(
        dataset_config=dataset_config,
        split_set=args.split,
        root_dir=args.dataset_dir,
    )
    print(f"Dataset length: {len(dataset)}")

    predictions_dir = Path(args.predictions_dir)

    # read prediction file
    predictions: dict[str, np.ndarray] = np.load(predictions_dir / f"{args.index}.npy", allow_pickle=True).item()

    # get point cloud
    point_cloud = dataset._read_lidar_data(name=dataset.filenames[args.index])
    ground_truth_bboxes = dataset._read_bboxes(name=dataset.filenames[args.index])
    ground_truth_corners = dataset_config.box_parametrization_to_corners_np(
        centers=ground_truth_bboxes[:, :3],
        sizes=ground_truth_bboxes[:, 3:6],
        angles=ground_truth_bboxes[:, 6],
    )

    # plot predictions
    fig = get_figure()
    plot_point_cloud(data=point_cloud, fig=fig)
    
    n = len(predictions["ids"])
    for idx in range(n):
        name = dataset_config.class2type[predictions["ids"][idx]]
        points_3d = predictions["corners"][idx]
        score = predictions["scores"][idx]
        plot_3d_bbox(points_3d=points_3d, name=name + "_P", fig=fig, extra_text=f"{round(score, 4)}")
    
    # plot ground truths
    for idx in range(len(ground_truth_bboxes)):
        name = dataset_config.class2type[ground_truth_bboxes[idx, 7]]
        plot_3d_bbox(points_3d=ground_truth_corners[idx], name=name, fig=fig)

    fig.show()
