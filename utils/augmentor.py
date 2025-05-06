"""3D Point Cloud Augmentor."""


import numpy as np
from utils.pc_util import rotz


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
        ) -> None:
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.translate = translate
        self.rotation = rotation
        self.scale = scale
        self.jitter = jitter
        self.dropout = dropout

    def __call__(self, point_cloud: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_cloud, bboxes = self.random_flip_x(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_flip_y(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_rotate(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_scaling(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_translate(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_jitter(point_cloud=point_cloud, bboxes=bboxes)
        point_cloud, bboxes = self.random_dropout(point_cloud=point_cloud, bboxes=bboxes)
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