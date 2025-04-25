"""Calibration Utils."""

import numpy as np
import traceback

from utils.matrix import inverse_rigid_transformation


class Calibration:

    def __init__(
        self,
        path: str,
    ) -> None:
        """Initialize Calibration.

        Args:
            path (str): Path to the calibration file
        """
        calibs = self._read_calib_file(path=path)
        # load neccessary data
        # projection Matrix: Rectified -> Image2
        self.P = np.reshape(calibs["P2"], [3, 4])
        # velodyne to reference camera - 0
        self.V2C = np.reshape(calibs["Tr_velo_to_cam"], [3, 4])
        # reference camera to velodyne
        self.C2V = inverse_rigid_transformation(tr=self.V2C)
        # rotation from reference camera to rectified camera
        self.R0 = np.reshape(calibs["R0_rect"], [3, 3])

    def _read_calib_file(self, path: str) -> dict[str, np.ndarray]:
        """Read Calibration text file.

        Args:
            path (str): File path

        Returns:
            dict[str, np.ndarray]: Calibration data
        """
        calibs = {}
        try:
            with open(path) as file:
                for line in file.readlines():
                    line = line.rstrip()
                    if len(line):
                        key, value = line.split(":", 1)
                        try:
                            calibs[key] = np.array(
                                [float(x) for x in value.split()])
                        except Exception:
                            pass
        except Exception as ex:
            error_str = "".join(
                traceback.format_exception(None, ex, ex.__traceback__))
            print(f"error loading calibration file, {path}, {error_str}")
        return calibs
    
    def cart2hom(self, points_3d: np.ndarray) -> np.ndarray:
        """Convert 3D Points from Cartesian to Homogeneous.

        Args:
            points_3d (np.ndarray): Cartesian Coordinates

        Returns:
            np.ndarray: Homogeneous Coordinates
        """
        return np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    def project_rect_to_ref(self, points_3d: np.ndarray) -> np.ndarray:
        """Proejct points from rectified space to reference camera space.

        Args:
            points_3d (np.ndarray): 3D Points

        Returns:
            np.ndarray: 3D points in reference camera space
        """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(points_3d)))
    
    def project_ref_to_velo(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from reference camera space to velodyne space.

        Args:
            points_3d (np.ndarray): 3D Points in reference camera space

        Returns:
            np.ndarray: 3D Points in velodyne space
        """
        return np.dot(self.cart2hom(points_3d), np.transpose(self.C2V))
    
    def project_rect_to_velo(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from rectified image space to velodyne space.

        Args:
            points_3d (np.ndarray): 3D points in rectified image space

        Returns:
            np.ndarray: 3D Points in velodyne space
        """
        points_3d = self.project_rect_to_ref(points_3d=points_3d)
        return self.project_ref_to_velo(points_3d=points_3d)
    
    def project_ref_to_rect(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from reference camera space to rectified image space.

        Args:
            points_3d (np.ndarray): 3D points in reference camera space

        Returns:
            np.ndarray: 3D Points in rectified image space
        """
        return np.transpose(np.dot(self.R0, np.transpose(points_3d)))
    
    def project_velo_to_ref(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from velodyne space to reference camera space.

        Args:
            points_3d (np.ndarray): 3D points in velodyne space

        Returns:
            np.ndarray: 3D Points in refernce camera space
        """
        points_3d = self.cart2hom(points_3d)  # nx4
        return np.dot(points_3d, np.transpose(self.V2C))
    
    def project_velo_to_rect(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from velodyne space to rectified image space.

        Args:
            points_3d (np.ndarray): 3D points in velodyne space

        Returns:
            np.ndarray: 3D Points in rectified image space
        """
        points_3d = self.project_velo_to_ref(points_3d)
        return self.project_ref_to_rect(points_3d)
    
    def project_rect_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from rectified image space to image space.

        Args:
            points_3d (np.ndarray): 3D points in rectified image space

        Returns:
            np.ndarray: 2D Points in image_2 space
        """
        points_3d = self.cart2hom(points_3d)
        points_3d = np.dot(points_3d, np.transpose(self.P))  # nx3
        points_3d[:, 0] /= points_3d[:, 2]
        points_3d[:, 1] /= points_3d[:, 2]
        return points_3d[:, 0:2]
    
    def project_velo_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """Project points from velodyne space to image.

        Args:
            points_3d (np.ndarray): 3D points in velodyne space

        Returns:
            np.ndarray: 2D Points in image_2 space
        """
        points_3d = self.project_velo_to_rect(points_3d)
        return self.project_rect_to_image(points_3d)
