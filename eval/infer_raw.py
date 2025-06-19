"""Infer on Raw KITTI dataset."""

import sys
sys.path.append("../")

from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from utils.calibration import Calibration
from utils.ground_removal import Processor as GroundRemover
from utils.ap_calculator import parse_predictions, get_ap_config_dict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_point_cloud(path: Path) -> np.ndarray:
    """Read Point Cloud File.

    Args:
        path (Path): Path to the point cloud file

    Returns:
        np.ndarray: Point Cloud Data
    """
    return np.fromfile(str(path), dtype=np.float32).reshape((-1, 4))


def read_image(path: Path) -> np.ndarray:
    """Read Image File.

    Args:
        path (Path): Path to image.

    Returns:
        np.ndarray: Image.
    """
    return np.array(Image.open(path).convert("RGB"))


def prepare_calibration_file(root_dir: Path) -> Path:
    """Prepare the Calibration file in required format.

    Args:
        root_dir (Path): Root directory of data

    Returns:
        Path: New Calibration file path.
    """
    cam_to_cam_path = root_dir.parent / "calib_cam_to_cam.txt"
    velo_to_cam_path = root_dir.parent / "calib_velo_to_cam.txt"

    # Filter the required fields and add it to the new calibration file
    fields_mapping = {
        "R_rect_00": "R0_rect",
        "P_rect_02": "P2",
    }

    # Get the fields from cam_to_cam
    calibration_txt = ""
    with open(cam_to_cam_path) as file:
        for line in file.readlines():
            line = line.rstrip()
            if len(line):
                key, value = line.split(":", 1)
                if key in fields_mapping:
                    calibration_txt += fields_mapping[key] + ":" + value + "\n"
    
    # Get the matrix from velo_to_cam
    tr_velo_to_cam = np.zeros((3, 4), dtype=np.float32)
    with open(velo_to_cam_path) as file:
        for line in file.readlines():
            line = line.rstrip()
            if len(line):
                key, value = line.split(":", 1)
                if key == "R":
                    tr_velo_to_cam[:, :3] = np.array([float(x) for x in value.split()]).reshape((3, 3))
                if key == "T":
                    tr_velo_to_cam[:, 3] = np.array([float(x) for x in value.split()])
    calibration_txt += "Tr_velo_to_cam: " + " ".join(tr_velo_to_cam.astype("str").flatten().tolist()) + "\n"

    output_path = root_dir / "calib.txt"
    with open(output_path, "w") as file:
        file.write(calibration_txt)
    
    return output_path


def preprocess_point_cloud(
    point_cloud: np.ndarray, 
    ground_remover: GroundRemover,
    radius: float = 15,
) -> np.ndarray:
    # Select XYZ
    point_cloud = point_cloud[:, :3]
    # Ground Points Remvoer
    point_cloud = ground_remover(point_cloud)
    # Filter Based on the radius
    distances = np.linalg.norm(point_cloud[:, :2], ord=2, axis=1)
    point_cloud = point_cloud[(distances >= 1) & (distances <= radius)]
    # Check if the point cloud is invalid
    invalid = False
    if point_cloud.shape[0] == 0:
        invalid = True
    else:
        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)
        size = point_cloud_dims_max - point_cloud_dims_min
        invalid = np.any(size == 0)
    if invalid:
        point_cloud = np.array([[0, 0, 0], [1, 1, 1]], dtype=point_cloud.dtype)
    return point_cloud


def divide_point_cloud(
    point_cloud: np.ndarray, 
    image: np.ndarray, 
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    # Select XYZ
    point_cloud = point_cloud[:, :3]
    segment_ids = np.zeros((point_cloud.shape[0],))
    # Project points to 2D
    points_2d = calibration.project_velo_to_image(points_3d=point_cloud[:, :3])
    # Segment 1
    xmin, xmax, ymin, ymax = 0, image.shape[1], 0, image.shape[0]
    clip_distance = 0
    index_1 = (
        (points_2d[:, 0] < xmax)
        & (points_2d[:, 0] >= xmin)
        & (points_2d[:, 1] < ymax)
        & (points_2d[:, 1] >= ymin)
        & (point_cloud[:, 0] > clip_distance)
    )
    segment_ids[index_1] = 1
    # Segment 2
    xmin, xmax, ymin, ymax = 0, image.shape[1], 0, image.shape[0]
    clip_distance = 0
    index_2 = (
        (points_2d[:, 0] < xmax)
        & (points_2d[:, 0] >= xmin)
        & (points_2d[:, 1] < ymax)
        & (points_2d[:, 1] >= ymin)
        & (point_cloud[:, 0] <= clip_distance)
    )
    segment_ids[index_2] = 2
    # Segment 3
    clip_distance = 0
    index_3 = (
        (segment_ids == 0)
        & (point_cloud[:, 1] > clip_distance)
    )
    segment_ids[index_3] = 3
    # Segment 3
    xmin, xmax, ymin, ymax = 0, image.shape[1], 0, image.shape[0]
    clip_distance = 0
    index_4 = (
        (segment_ids == 0)
        & (point_cloud[:, 1] <= clip_distance)
    )
    segment_ids[index_4] = 4
    return point_cloud, segment_ids


def load_model(model_path: Path) -> torch.nn.Module:
    model = torch.load(str(model_path), weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model


def process_outputs(point_clouds: np.ndarray, outputs: dict[str, torch.Tensor]) -> list:
    predictions = parse_predictions(
        predicted_boxes=outputs["box_corners"],
        sem_cls_probs=outputs["sem_cls_prob"],
        objectness_probs=outputs["objectness_prob"],
        point_cloud=point_clouds,
        config_dict=get_ap_config_dict(remove_empty_box=True),
    )[0]
    n = len(predictions)
    output = {
        "ids": np.zeros((n,), dtype=np.uint32),
        "corners": np.zeros((n, 8, 3), dtype=np.float32),
        "scores": np.zeros((n,), dtype=np.float32),
        "ious": np.zeros((n,), dtype=np.float32),
    }
    for jdx, (id, corners, score, iou) in enumerate(predictions):
        output["ids"][jdx] = id
        output["corners"][jdx] = corners
        output["scores"][jdx] = score
        output["ious"][jdx] = iou
    return output


def run_raw_inference(
    root_dir: Path,
    output_dir: Path,
    model_path: Path,
    radius: float = 15,
    three60: bool = False,
) -> None:
    
    point_cloud_dir = root_dir / "velodyne_points/data"
    images = root_dir / "image_02/data"

    # Create Output Dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare calibration file
    calibration_path = prepare_calibration_file(root_dir=root_dir)
    calibration = Calibration(path=str(calibration_path))

    # Ground Points Remover
    ground_remover = GroundRemover(
        n_segments=70,
        n_bins=80,
        line_search_angle=0.3,
        max_dist_to_line=0.15,
        sensor_height=-1.73,
        max_start_height=0.25,
        long_threshold=8,
        r_max=radius,
    )

    # Model
    model = load_model(model_path=model_path)

    for path in tqdm(sorted(point_cloud_dir.glob("*.bin"))):
        
        name = path.stem
        point_cloud = read_point_cloud(path=path)
        image = read_image(path=images / f"{name}.png")

        # Preprocess the Point cloud
        point_cloud = preprocess_point_cloud(
            point_cloud=point_cloud, 
            ground_remover=ground_remover, 
            radius=radius
        )
        point_cloud, segment_ids = divide_point_cloud(point_cloud=point_cloud, image=image, calibration=calibration)

        aggregated_outputs = []
        for segment_id in np.unique(segment_ids):

            if not three60 and segment_id != 1:
                continue

            _point_cloud = point_cloud[segment_ids == segment_id]

            point_cloud_dims_min = _point_cloud.min(axis=0)
            point_cloud_dims_max = _point_cloud.max(axis=0)

            # Prepare Model input
            inputs = {
                "point_clouds": torch.tensor(_point_cloud[None, ...]),
                "point_cloud_dims_min": torch.tensor(point_cloud_dims_min[None, ...]),
                "point_cloud_dims_max": torch.tensor(point_cloud_dims_max[None, ...]),
            }
            for key in inputs:
                inputs[key] = inputs[key].to(DEVICE)

            # Run Inference
            outputs = model(inputs)["outputs"]
            outputs = process_outputs(point_clouds=inputs["point_clouds"], outputs=outputs)
            aggregated_outputs.append(outputs)

        # Aggregate the dictionary values
        outputs = aggregated_outputs[0]
        for idx in range(1, len(aggregated_outputs)):
            for key in outputs:
                outputs[key] = np.r_[outputs[key], aggregated_outputs[idx][key]]

        # Save Predictions
        np.save(str(output_dir / f"{name}.npy"), outputs)


if __name__ == "__main__":

    parser = ArgumentParser(description="KITTI Raw Data Inference")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=Path,
        help="KITTI continuous raw data path",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Ouput Predictions Directory",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model_path",
        type=Path,
        help="Model Path",
        required=True,
    )

    parser.add_argument(
        "--radius",
        type=float,
        help="Scene Radius",
        default=15,
    )

    parser.add_argument(
        "--three60",
        help="Enable 360 degree predictions",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    run_raw_inference(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        radius=args.radius,
        three60=args.three60,
    )