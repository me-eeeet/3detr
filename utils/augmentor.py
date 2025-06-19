"""3D Point Cloud Augmentor."""


import numpy as np
from utils.pc_util import rotz
from utils.random_cuboid import RandomCuboid
from utils.box_util import extract_pc_in_box3d, box3d_iou


class PointAugmentor:

    def __init__(
            self,
            flip_x: float = 0.5,
            flip_y: float = 0.5,
            translate: float = 0.1,
            rotation: float = np.pi / 6,
            scale: tuple[float, float] = (0.85, 1.15),
            jitter: float = 0.01,
            dropout: float = 0.0,
            cuboid: tuple[float, float, float] = [0.5, 1.0],
        ) -> None:
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.translate = translate
        self.rotation = rotation
        self.scale = scale
        self.jitter = jitter
        self.dropout = dropout
        self.cuboid = RandomCuboid(
            min_points=3072,
            aspect=0.5,
            min_crop=cuboid[0],
            max_crop=cuboid[1],
        )

    def __call__(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_cloud, bboxes = self.random_flip_x(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_flip_y(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_rotate(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_scaling(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_translate(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_jitter(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_dropout(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_cuboid(point_cloud=point_cloud, bboxes=bboxes)
        return point_cloud, bboxes

    def random_flip_x(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() <= self.flip_x:
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            bboxes[:, 0] = -1 * bboxes[:, 0]
            bboxes[:, 6] = -1 * bboxes[:, 6]
        return point_cloud, bboxes
    
    def random_flip_y(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() <= self.flip_y:
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            bboxes[:, 1] = -1 * bboxes[:, 1]
            bboxes[:, 6] = np.pi - bboxes[:, 6]
        return point_cloud, bboxes

    def random_rotate(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot_angle = np.random.uniform(low=-self.rotation, high=self.rotation)
        rot_mat = rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
        bboxes[:, 6] += rot_angle
        return point_cloud, bboxes

    def random_scaling(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale_ratio = np.random.uniform(low=self.scale[0], high=self.scale[1])
        scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
        point_cloud[:, 0:3] *= scale_ratio
        bboxes[:, 0:3] *= scale_ratio
        bboxes[:, 3:6] *= scale_ratio
        return point_cloud, bboxes
    
    def random_translate(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_coords = point_cloud.min(axis=0)
        max_coords = point_cloud.max(axis=0)
        size = max_coords - min_coords
        translation = size * np.random.uniform(-self.translate, self.translate, size=(1, 3))
        point_cloud[:, 0:3] += translation
        bboxes[:, 0:3] += translation
        return point_cloud, bboxes
    
    def random_jitter(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jitter = np.clip(self.jitter * np.random.randn(*point_cloud[:, 0:3].shape), -0.05, 0.05)
        point_cloud[:, 0:3] += jitter
        return point_cloud, bboxes
    
    def random_dropout(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_points = point_cloud.shape[0]
        keep_mask = np.random.rand(n_points) > self.dropout
        if np.sum(keep_mask) == 0:
            keep_mask[np.random.randint(n_points)] = True
        # pad with zeros to maintain shape
        point_cloud = point_cloud[keep_mask]
        return point_cloud, bboxes
    
    def random_cuboid(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_cloud, bboxes, _ = self.cuboid(point_cloud=point_cloud, target_boxes=bboxes)
        return point_cloud, bboxes
    

class GTSampler:

    def __init__(
        self, 
        dataset_config, 
        p: float = 0.5, 
        max_iou: float = 0.1, 
        max_points: float = 10,
    ) -> None:
        self.dataset_config = dataset_config
        self.p = p
        self.max_iou = max_iou
        self.max_points = max_points

    def __call__(
        self, 
        point_cloud: np.ndarray, 
        bboxes: np.ndarray, 
        point_cloud_b: np.ndarray, 
        bboxes_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        min_coords = point_cloud.min(axis=0)
        max_coords = point_cloud.max(axis=0)
        corners1 = self.dataset_config.box_parametrization_to_corners_np(centers=bboxes[:, :3], sizes=bboxes[:, 3:6], angles=bboxes[:, 6])
        for i in range(bboxes_b.shape[0]):
            if np.random.random() < self.p:
                bbox = bboxes_b[i]
                # corners
                corners = self.dataset_config.my_compute_box_3d(center=bbox[:3], size=bbox[3:6], angle=bbox[6])
                # get the points inside this bbox
                points_inside, _ = extract_pc_in_box3d(point_cloud_b, corners)
                # normalize the points
                points_inside[:, :3] -= bbox[:3]
                # randomly rotate the object
                rotation = np.random.uniform(-np.pi / 6, np.pi / 6)
                rot_mat = rotz(rotation)
                points_inside[:, 0:3] = np.dot(points_inside[:, 0:3], np.transpose(rot_mat))
                # paste the object
                tries = 5
                while tries > 0:
                    # find a random destination point
                    new_center = np.random.uniform(min_coords, max_coords)
                    # keep the z corrdinate same
                    new_center[2] = min_coords[2] + bbox[5] / 2
                    # corners
                    corners = self.dataset_config.my_compute_box_3d(center=new_center, size=bbox[3:6], angle=bbox[6]+rotation)
                    # check if there are points inside the bbox
                    indices = extract_pc_in_box3d(point_cloud, corners)[1]
                    if np.sum(indices) <= self.max_points:
                        # check it it's overlapping with other objects
                        for j in range(bboxes.shape[0]):
                            if box3d_iou(corners1=corners1[j], corners2=corners)[0] >= self.max_iou:
                                break
                        else:
                            points_inside[:, :3] += new_center
                            # add the points to the point cloud
                            point_cloud = np.r_[point_cloud, points_inside]
                            # add the bbox
                            bbox[:3] = new_center
                            bbox[6] += rotation
                            bboxes = np.r_[bboxes, bbox[None, ...]]
                            # update corners
                            corners1 = np.r_[corners1, corners[None, ...]]
                            break
                    tries -= 1
        return point_cloud, bboxes
    