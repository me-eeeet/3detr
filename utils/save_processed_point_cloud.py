import sys
sys.path.append("../")

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from datasets.kitti_copy import KITTIDetectionDataset, KITTIDatasetConfig
from utils.ground_removal import Processor as GroundRemover


if __name__ == "__main__":

    parser = ArgumentParser(description="KITTI Data Visualizer")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        help="KITTI data root directory path",
        required=True,
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
        "-o",
        "--output_dir",
        type=Path,
        help="Index of the point cloud",
        default=0,
    )

    args = parser.parse_args()

    config = KITTIDatasetConfig()
    dataset = KITTIDetectionDataset(
        dataset_config=config,
        split_set=args.split,
        root_dir=args.root_dir,
        augment=False,
    )

    ground_remover = GroundRemover(
        n_segments=70,
        n_bins=80,
        line_search_angle=0.3,
        max_dist_to_line=0.15,
        sensor_height=-1.73,
        max_start_height=0.25,
        long_threshold=8,
        r_max=dataset.radius,
    )

    # create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # read point cloud data, process it and save it
    for idx in tqdm(range(len(dataset))):
        filename = dataset.filenames[idx]
        point_cloud = dataset._read_lidar_data(name=filename)[:, :3]
        calibration = dataset._read_calibration(name=filename)
        # Filter out points outside of the FOV
        point_cloud = dataset._filter_fov_points(points_3d=point_cloud, calibration=calibration)
        # remove ground points
        point_cloud = ground_remover(point_cloud)
        # Filter Based on the radius
        distances = np.linalg.norm(point_cloud[:, :2], ord=2, axis=1)
        point_cloud = point_cloud[distances <= dataset.radius]
        # Save the point cloud
        point_cloud.astype(np.float32).tofile(str(args.output_dir / f"{filename}.bin"))