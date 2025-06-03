"""Plot Predictions."""

import sys
sys.path.append("../")
sys.path.append("../third_party/pointnet2")

import numpy as np
from pathlib import Path
import torch
import plotly.graph_objects as go

from datasets.kitti_copy import KITTIDatasetConfig, KITTIDetectionDataset
from utils.plotting import get_figure, plot_point_cloud, plot_3d_bbox
from third_party.pointnet2.pointnet2_utils import furthest_point_sample


def get_plot(
    index: int,
    predictions_dir: str,
    n_points: float,
    dataset_config: KITTIDatasetConfig,
    dataset: KITTIDetectionDataset,
) -> go.Figure:
    
    predictions_dir = Path(predictions_dir)
    raw_point_cloud_dir = Path("/common/dataset/kitti/data_object_velodyne5/training/velodyne")

    # read prediction file
    predictions: dict[str, np.ndarray] = np.load(predictions_dir / f"{index}.npy", allow_pickle=True).item()
    filename = dataset.filenames[index]

    data = dataset.__getitem__(index=index)
    raw_point_cloud = np.fromfile(str(raw_point_cloud_dir / f"{filename}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
    point_cloud = data["point_clouds"]

    if n_points:
        batch_point_cloud = torch.tensor(point_cloud[:, :3])[None, ...]
        batch_point_cloud = batch_point_cloud.cuda()
        indexes = furthest_point_sample(batch_point_cloud, n_points)
        indexes = indexes.cpu().numpy()[0]
        point_cloud = point_cloud[indexes]
    
    ground_truth_corners = data["gt_box_corners"]
    ground_truth_labels = data["gt_box_sem_cls_label"]
    ground_truth_present = data["gt_box_present"]
    ground_truth_difficulty = data["gt_difficulty"]

    # plot predictions
    fig = get_figure(rows=1, cols=2, titles=["GroundTruth", "Predictions"])
    plot_point_cloud(data=raw_point_cloud, name=dataset.filenames[index], fig=fig, row=1, col=1)
    plot_point_cloud(data=point_cloud, name=dataset.filenames[index], fig=fig, row=1, col=2)
    
    n = len(predictions["ids"])
    for idx in range(n):
        name = dataset_config.class2type[predictions["ids"][idx]]
        points_3d = predictions["corners"][idx]
        score = predictions["scores"][idx]
        iou = predictions["ious"][idx]
        plot_3d_bbox(points_3d=points_3d, name=name, fig=fig, extra_text=f"{score:0.2f}_{iou:0.2f}", row=1, col=2)
    
    # plot ground truths
    for idx in range(len(ground_truth_corners)):
        if ground_truth_present[idx]:
            name = dataset_config.class2type[ground_truth_labels[idx]]
            difficulty = dataset_config.difficulty_map[ground_truth_difficulty[idx]]
            plot_3d_bbox(points_3d=ground_truth_corners[idx], name=name, fig=fig, extra_text=difficulty, row=1, col=1)
        else:
            break

    return fig
