"""Visualize 3D Point Cloud with Annotations."""

from argparse import ArgumentParser

from datasets import KITTIDetectionDataset, KITTIDatasetConfig


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
        "-i",
        "--idx",
        type=int,
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
    print(f"Dataset Length: {len(dataset)}")
    dataset.plot_data(idx=args.idx)