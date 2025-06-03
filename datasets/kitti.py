"""KITTI Dataset & Config."""

import traceback
from pathlib import Path


import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from utils.calibration import Calibration
from utils.object3d import Object3D
from utils.plotting import get_figure, plot_point_cloud, plot_3d_bbox
import utils.pc_util as pc_util
from utils.box_util import rotz_batch, rotz_batch_tensor, filter_bboxes_low_points
from utils.augmentor import PointAugmentor, GTSampler


class KITTIDatasetConfig(object):

    def __init__(self) -> None:
        self.type2class =  { 
            'Car': 0, 
            'Pedestrian': 1,
            'Cyclist': 2,
        }
        self.ignored_semcls = {
            'Van': 3, 
            'Truck': 4,
            'Person_sitting': 5,
            'Tram': 6, 
            'Misc': 7,
            "DontCare": 8,
        }
        self.difficulty_map = {
            0: "easy",
            1: "medium",
            2: "hard",
            -1: "none"
        }
        self.num_semcls = len(self.type2class)
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.num_angle_bin = 12
        self.max_num_obj = 64

    def angle2class(self, angle: float) -> tuple[int, float]:
        """Quantize the angles into N bins from [0, 2π) and calculate the quantization residual.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that 
        angle = class*(2pi/N) + residual

        Args:
            angle (float): Angle in radians. Range: -pi to pi

        Returns:
            tuple[int, float]: Quantized angle & Residual
        """
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / self.num_angle_bin
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls: int, residual: float, to_label_format: bool = True) -> float:
        """Inverse angle2class.

        Args:
            pred_cls (int): Bin ID.
            residual (float): Angle residual.
            to_label_format (bool, optional): -pi to pi range. Defaults to True.

        Returns:
            float: Angle in radians.
        """
        angle_per_class = 2 * np.pi / self.num_angle_bin
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(
            self,
            pred_cls: np.ndarray,
            residual: np.ndarray,
            to_label_format: bool = True,
        ) -> np.ndarray:
        """Class to Angle for a batch.

        Args:
            pred_cls (np.ndarray): Predicted Bin IDs.
            residual (np.ndarray): Predicted residual values.
            to_label_format (bool, optional): -pi to pi range. Defaults to True.

        Returns:
            np.ndarray: Angles in radian
        """
        angle_per_class = 2 * np.pi / self.num_angle_bin
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)
    
    def box_parametrization_to_corners(self, center, box_size, angle):
        assert isinstance(box_size, torch.Tensor)
        assert isinstance(angle, torch.Tensor)
        assert isinstance(center, torch.Tensor)

        reshape_final = False
        if angle.ndim == 2:
            assert box_size.ndim == 3
            assert center.ndim == 3
            bsize = box_size.shape[0]
            nprop = box_size.shape[1]
            box_size = box_size.reshape(-1, box_size.shape[-1])
            angle = angle.reshape(-1)
            center = center.reshape(-1, 3)
            reshape_final = True

        input_shape = angle.shape
        R = rotz_batch_tensor(angle)
        w = torch.unsqueeze(box_size[..., 0], -1)  # [x1,...,xn,1]
        l = torch.unsqueeze(box_size[..., 1], -1)
        h = torch.unsqueeze(box_size[..., 2], -1)
        corners_3d = torch.zeros(
            tuple(list(input_shape) + [8, 3]), device=box_size.device, dtype=torch.float32
        )
        corners_3d[..., :, 0] = torch.cat(
            (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
        )
        corners_3d[..., :, 1] = torch.cat(
            (-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2), -1
        )
        corners_3d[..., :, 2] = torch.cat(
            (-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2), -1
        )
        tlist = [i for i in range(len(input_shape))]
        tlist += [len(input_shape) + 1, len(input_shape)]
        corners_3d = torch.matmul(corners_3d, R.permute(tlist))
        corners_3d += torch.unsqueeze(center, -2)
        if reshape_final:
            corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
        return corners_3d

    def box_parametrization_to_corners_np(self, centers: np.ndarray, sizes: np.ndarray, angles: np.ndarray) -> np.ndarray:
        input_shape = angles.shape
        R = rotz_batch(angles)
        w = np.expand_dims(sizes[..., 0], -1)  # [x1,...,xn,1]
        l = np.expand_dims(sizes[..., 1], -1)
        h = np.expand_dims(sizes[..., 2], -1)
        corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
        corners_3d[..., :, 0] = np.concatenate(
            (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
        )
        corners_3d[..., :, 1] = np.concatenate(
            (-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2), -1
        )
        corners_3d[..., :, 2] = np.concatenate(
            (-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2), -1
        )
        tlist = [i for i in range(len(input_shape))]
        tlist += [len(input_shape) + 1, len(input_shape)]
        corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
        corners_3d += np.expand_dims(centers, -2)
        return corners_3d
    
    def my_compute_box_3d(self, center, size, angle):
        R = pc_util.rotz(angle)
        w, l, h = size / 2
        x_corners = [w, -w, -w, w, w, -w, -w, w]
        y_corners = [-l, -l, l, l, -l, -l, l, l]
        z_corners = [-h, -h, -h, -h, h, h, h, h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)


class KITTIDetectionDataset(Dataset):

    def __init__(
        self,
        dataset_config: KITTIDatasetConfig,
        split_set: str = "train",
        root_dir: str = None,
        radius: float = 15,
        num_points: int = 15000,
        augment: bool = False,
        use_random_cuboid: bool = True,
        random_cuboid_min_points: int = 30000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert split_set in ["train", "val", "trainval", "test"]
        split_dir = "testing" if split_set in {"test"} else "training"

        self.dataset_config = dataset_config
        self.radius = radius
        self.num_points = num_points
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_min_points = random_cuboid_min_points
        self.augmentor = PointAugmentor()
        self.gt_sampler = GTSampler(dataset_config=dataset_config, p=0.7)
        
        # load filenames
        with open(str(Path(root_dir) / "sets" / f"{split_set}.txt")) as file:
            self.filenames = sorted(list(map(lambda x: x.strip(), file.readlines())))

        root_dir: Path = Path(root_dir) / split_dir
        temp_root_dir: Path = Path("/common/dataset/kitti/data_object_velodyne5") / split_dir
        # initialize subfolder paths
        self.calib_dir = root_dir / "calib"
        self.image_2_dir = root_dir / "image_2"
        self.label_2_dir = root_dir / "label_2"
        self.bboxes_dir = root_dir / "bboxes_2"
        self.velodyne_dir = root_dir / "velodyne_processed"
        
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 16

    def __len__(self) -> int:
        """Return length of the data."""
        return int(len(self.filenames) * 1.0)

    def _read_calibration(self, name: str) -> Calibration:
        """Read Calibration Data.

        Args:
            name (str): Name of the file

        Returns:
            Calibration: Calibration Object
        """
        return Calibration(str(self.calib_dir / f"{name}.txt"))

    def _read_image(self, name: str) -> Image.Image:
        """Load Image.

        Args:
            name (str): Namae of the image

        Returns:
            np.ndarray: Image
        """
        return np.array(
            Image.open(str(self.image_2_dir / f"{name}.png")).convert("RGB"))

    def _read_label(self, name: str) -> list[Object3D]:
        """Read 3D objects from the label file.

        Args:
            name (str): Name of the file

        Returns:
            list[Object3D]: List of 3D Objects
        """
        objects = []
        path = str(self.label_2_dir / f"{name}.txt")
        with open(path) as file:
            for line in file.readlines():
                try:
                    line = line.rstrip()
                    if len(line):
                        object3d = Object3D(line=line)
                        if object3d.name not in self.dataset_config.ignored_semcls:
                            objects.append(object3d)
                except Exception as ex:
                    error_str = "".join(
                        traceback.format_exception(None, ex, ex.__traceback__))
                    print(
                        f"error reading label file, {path}, line, {line}, {error_str}"
                    )
        return objects

    def _read_lidar_data(self, name: str) -> np.ndarray:
        """Read Velodyne Lidar Point Cloud Data.

        Args:
            name (str): Name of the file

        Returns:
            np.ndarray: 3D Point Cloud
        """
        return np.fromfile(str(self.velodyne_dir / f"{name}.bin"),
                           dtype=np.float32).reshape((-1, 3))
    
    def _read_bboxes(self, name: str) -> np.ndarray:
        """Read the processes BBoxes in Velodyne space.

        Args:
            name (str): Name of the file

        Returns:
            np.ndarray: BBoxes
        """
        return np.load(self.bboxes_dir / f"{name}.npy")

    def _read_data(self, idx: int, raw: bool = False) -> dict:
        """Read KITTI datapoint.

        Args:
            idx (int): Index of the file
            raw (bool): Whether to read raw data or processed

        Returns:
            dict: KITTI Data point 
        """
        assert idx < len(
            self.filenames
        ), f"index out of range {idx} >= {len(self.filenames)}"
        name = self.filenames[idx]
        return {
            "calibration": self._read_calibration(name=name),
            "objects": self._read_label(name=name) if raw else None,
            "bboxes": self._read_bboxes(name=name) if not raw else None,
            "lidar": self._read_lidar_data(name=name),
        }
    
    def _preprocess_3d_object(self, object: Object3D, calibration: Calibration) -> np.ndarray:
        """Preprocess the Raw 3D object points and project them to Velodyne space.

        Args:
            object (Object3D): 3D Object instance
            calibration (Calibration): Calibration instance

        Returns:
            np.ndarray: 3D Points in Velodyne space
        """
        points_3d_rect = object.get_3d_bbox()
        points_3d_velo = calibration.project_rect_to_velo(points_3d=points_3d_rect)
        return points_3d_velo
    
    def _filter_fov_points(self, points_3d: np.ndarray, calibration: Calibration) -> np.ndarray:
        """Filter out points outside of Image FOV.

        Args:
            points_3d (np.ndarray): 3D Point cloud points

        Returns:
            np.ndarray: Filtered Point cloud points
        """
        xmin, xmax, ymin, ymax = 0, 1242, 0, 375
        clip_distance = 0
        points_2d = calibration.project_velo_to_image(points_3d=points_3d)
        index = (
            (points_2d[:, 0] < xmax)
            & (points_2d[:, 0] >= xmin)
            & (points_2d[:, 1] < ymax)
            & (points_2d[:, 1] >= ymin)
            & (points_3d[:, 0] > clip_distance)
        )
        return points_3d[index]
    
    def _add_difficulty(self, objects: list[Object3D]) -> list[Object3D]:
        min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
        max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
        max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
        n_objects = len(objects)
        easy_mask = np.ones((n_objects, ), dtype=np.bool)
        moderate_mask = np.ones((n_objects, ), dtype=np.bool)
        hard_mask = np.ones((n_objects, ), dtype=np.bool)
        for idx, object3d in enumerate(objects):
            o, h, t = object3d.occlusion, object3d.box2d[3] - object3d.box2d[1], object3d.truncation
            if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
                easy_mask[idx] = False
            if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
                moderate_mask[idx] = False
            if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
                hard_mask[idx] = False
        is_easy = easy_mask
        is_moderate = np.logical_xor(easy_mask, moderate_mask)
        is_hard = np.logical_xor(hard_mask, moderate_mask)
        for idx, object3d in enumerate(objects):
            if is_easy[idx]:
                object3d.difficulty = 0
            elif is_moderate[idx]:
                object3d.difficulty = 1
            elif is_hard[idx]:
                object3d.difficulty = 2
        return objects

    def _is_invalid_point_cloud(self, point_cloud: np.ndarray) -> bool:
        if point_cloud.shape[0] == 0:
            return True
        else:
            point_cloud_dims_min = point_cloud.min(axis=0)
            point_cloud_dims_max = point_cloud.max(axis=0)
            size = point_cloud_dims_max - point_cloud_dims_min
            return np.any(size == 0)
    
    def _filter_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        bboxes = bboxes[~np.isin(bboxes[:, 7], list(self.dataset_config.ignored_semcls.values()))]
        # bboxes = bboxes[np.isin(bboxes[:, 8], [0])] # Easy bboxes
        # bboxes in the radius
        distances = np.linalg.norm(bboxes[:, :2], ord=2, axis=1)
        return bboxes[distances <= self.radius]
    
    def __getitem__(self, index: int):
        
        # read data from files
        data = self._read_data(idx=index)

        point_cloud = data["lidar"][:, :3]

        # Add random points
        if self._is_invalid_point_cloud(point_cloud=point_cloud):
            point_cloud = np.array([[0, 0, 0], [1, 1, 1]], dtype=point_cloud.dtype)
            
        # filter unnecessary bboxes
        bboxes: np.ndarray = data["bboxes"]
        bboxes = self._filter_bboxes(bboxes=bboxes)

        # augmentations
        if self.augment:
            point_cloud, bboxes = self.augmentor(point_cloud=point_cloud, bboxes=bboxes)
            # get random data point
            filename2 = np.random.choice(self.filenames)
            point_cloud2 = self._read_lidar_data(name=filename2)[:, :3]
            bboxes2 = self._filter_bboxes(self._read_bboxes(name=filename2))
            # gt sampler
            point_cloud, bboxes = self.gt_sampler(point_cloud=point_cloud, bboxes=bboxes, point_cloud_b=point_cloud2, bboxes_b=bboxes2)
        
        # remove bboxes not having minimum points
        corners = self.dataset_config.box_parametrization_to_corners_np(centers=bboxes[:, :3], sizes=bboxes[:, 3:6], angles=bboxes[:, 6])
        mask = filter_bboxes_low_points(point_cloud=point_cloud, corners=corners, min_points=100)
        bboxes = bboxes[mask]
        n_bboxes = bboxes.shape[0]

        # ------------------------------- LABELS ------------------------------
        box_centers = np.zeros((self.max_num_obj, 3))
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        difficulty = np.zeros((self.max_num_obj,))
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_mask = np.zeros((self.max_num_obj))
        target_bboxes_mask[:n_bboxes] = 1

        for i in range(min(n_bboxes, self.max_num_obj)):
            bbox = bboxes[i]
            target_bboxes_semcls[i] = bbox[7]
            raw_angles[i] = bbox[6]
            raw_sizes[i, :] = bbox[3:6]
            angle_classes[i], angle_residuals[i] = self.dataset_config.angle2class(bbox[6])
            corners_3d = self.dataset_config.my_compute_box_3d(bbox[:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            box_centers[i] = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                ]
            )
            difficulty[i] = bbox[8]

        # Randomly sample point clound points
        point_cloud = pc_util.random_sampling(
            point_cloud, self.num_points if self.num_points > 0 else len(point_cloud), return_choices=False
        )[:, :3]

        # Normalize the boxes (Size & Center)
        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = pc_util.scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers_normalized = pc_util.shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        # Calculate the box corners in Velodyne Space (To verify GT sizes and angles)
        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            centers=box_centers[None, ...],
            sizes=raw_sizes.astype(np.float32)[None, ...],
            angles=raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(np.float32)
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(index).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["gt_difficulty"] = difficulty.astype(np.int64)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        return ret_dict
    
    def plot_data(self, idx: int) -> None:
        """Plot 3D Point Cloud & Annotations.

        Args:
            idx (int): Index of the data file
        """
        data = self.__getitem__(index=idx)
        # create a figure
        fig = get_figure()
        # plot 3D point cloud
        print(f"Lidar Points Shape: {data['point_clouds'].shape}")
        plot_point_cloud(data=data["point_clouds"], name=self.filenames[idx], fig=fig)
        # plot annotations
        n_objects = int(np.sum(data['gt_box_present']))
        print(f"Number of Objects: {n_objects}")
        for i in range(n_objects):
            name = self.dataset_config.class2type[data["gt_box_sem_cls_label"][i]]
            points_3d = data["gt_box_corners"][i]
            print(f"Object: {name}\nPoints: {points_3d}")
            plot_3d_bbox(points_3d=points_3d, name=name, fig=fig, extra_text=self.dataset_config.difficulty_map[data["gt_difficulty"][i]])
        fig.show()

    def convert(self) -> None:
        """Convert annotation data from raw format (Camera Space) to Velodyne Space."""
        # create the output dir
        self.bboxes_dir.mkdir(parents=True, exist_ok=True)
        # iterate and save the 
        for index, filename in tqdm(enumerate(self.filenames)):
            data = self._read_data(idx=index, raw=True)
            # BBoxes format - cx, cy, cz, sx, sy, sz, rz, cls, difficulty
            bboxes = np.zeros((len(data["objects"]), 9), dtype=np.float32)
            # Add Difficulty
            data["objects"] = self._add_difficulty(objects=data["objects"])
            object3d: Object3D
            for i, object3d in enumerate(data["objects"]):
                corners_3d = self._preprocess_3d_object(object=object3d, calibration=data["calibration"])
                # centroids
                xmin = np.min(corners_3d[:, 0])
                ymin = np.min(corners_3d[:, 1])
                zmin = np.min(corners_3d[:, 2])
                xmax = np.max(corners_3d[:, 0])
                ymax = np.max(corners_3d[:, 1])
                zmax = np.max(corners_3d[:, 2])
                bboxes[i, :3] = [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                ]
                # sizes
                bboxes[i, 3:6] = [object3d.w, object3d.l, object3d.h]
                # yaw angle in velodyne space
                bboxes[i, 6] = -object3d.ry
                # class
                bboxes[i, 7] = self.dataset_config.type2class[object3d.name]
                # difficulty
                bboxes[i, 8] = object3d.difficulty
            # save the bboxes
            np.save(self.bboxes_dir / f"{filename}.npy", bboxes)
