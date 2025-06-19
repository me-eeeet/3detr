"""3D Object."""

import numpy as np
from utils.matrix import rot_y


class Object3D:

    def __init__(
        self,
        line: str,
    ) -> None:
        """Initialize 3D Object.

        Label Field Info: https://medium.com/data-science/kitti-coordinate-transformations-125094cd42fb

        Args:
            line (str): Label Line
        """
        values = line.split(" ")
        values[1:] = list(map(float, values[1:]))
        # store neccessary values
        self.name, self.truncation, self.occlusion, self.alpha = values[:4]
        # 2D Bounding box
        self.xmin, self.ymin, self.xmax, self.ymax = values[4:8]
        self.box2d = np.array(values[4:8])
        # 3D Bounding Box
        self.h, self.w, self.l = values[8:11]
        self.t = tuple(values[11:14])
        self.ry = values[14]
        # Difficulty
        self.difficulty = -1

    def get_3d_bbox(self) -> np.ndarray:
        """Return 3D BBox in rectified image coordinate space.

        Returns:
            np.ndarray: 3D BBox
        """
        # rotation matrix
        R = rot_y(self.ry)
        # translation matrix
        t = np.array(self.t).reshape((3, 1))
        # transformation matrix
        tr = np.hstack([R, t])
        # 3d bounding box corners
        x_corners = [
            self.l / 2, self.l / 2, -self.l / 2, -self.l / 2, 
            self.l / 2, self.l / 2, -self.l / 2, -self.l / 2
        ]
        y_corners = [0, 0, 0, 0, -self.h, -self.h, -self.h, -self.h]
        z_corners = [
            self.w / 2, -self.w / 2, -self.w / 2, self.w / 2, 
            self.w / 2, -self.w / 2, -self.w / 2, self.w / 2
        ]
        # rotate and translate 3d bounding box
        bbox_3d = np.transpose(
            np.dot(
                tr,
                np.vstack([
                    x_corners, y_corners, z_corners,
                    np.ones((len(x_corners), ))
                ])))
        return bbox_3d
