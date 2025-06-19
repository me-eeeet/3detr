"""Visualize 3D Point Cloud with Annotations."""

import sys
sys.path.append("../")

from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from datasets.kitti_copy import KITTIDetectionDataset, KITTIDatasetConfig
from utils.plotting import get_figure, plot_point_cloud, plot_3d_bbox
from utils.object3d import Object3D


def get_plot(
    index: int,
    dataset_config: KITTIDatasetConfig,
    dataset: KITTIDetectionDataset,
) -> go.Figure:
    
    raw_point_cloud_dir = Path("/common/dataset/kitti/data_object_velodyne5/training/velodyne")
    
    filename = dataset.filenames[index]
    data = dataset._read_data(idx=index, raw=True)

    raw_point_cloud = np.fromfile(str(raw_point_cloud_dir / f"{filename}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]

    # Add Difficulty
    data["objects"] = dataset._add_difficulty(objects=data["objects"])
    
    # plot predictions
    fig = get_figure(rows=1, cols=1)
    plot_point_cloud(data=raw_point_cloud, name=filename, fig=fig)

    object3d: Object3D
    for i, object3d in enumerate(data["objects"]):
        corners_3d = dataset._preprocess_3d_object(object=object3d, calibration=data["calibration"])
        difficulty = dataset_config.difficulty_map[object3d.difficulty]
        plot_3d_bbox(points_3d=corners_3d, name=object3d.name, fig=fig, extra_text=f"{difficulty}_T{object3d.truncation}_O{object3d.occlusion}")

    return fig