import math
import numpy as np

from pathlib import Path
from scipy.spatial.transform import Rotation as R

from .geometry import Pose
from .model import Model


class Camera(Model):
    DEFAULT_MODEL = Path(__file__).parent / "../obj/Camera.obj"

    def __init__(self, name: str,
                 pose: str | Pose = Pose(1.5, -1, 0.5, 90, 0, 0),
                 h_fov: float = 70,
                 res: str | tuple[int, int] = (1920, 1080),
                 near: float = 0.01,
                 far: float = 20,
                 focal: float = 0.01,
                 orthographic: bool = False):
        super().__init__(self.DEFAULT_MODEL, name, pose)
        self.h_fov: float = h_fov
        self.res: tuple[int, int] = eval(res, {}, {res: 'res'}) if isinstance(res, str) else res
        self.width: int = self.res[0]
        self.height: int = self.res[1]
        self.v_fov: float = h_fov * self.height / self.width
        self.near: float = near
        self.far: float = far
        self.f: float = focal
        self.orthographic: bool = orthographic
        self.sx = 2 * self.f * math.tan(math.pi * self.h_fov / (180 * 2)) / self.width
        self.sy = 2 * self.f * math.tan(math.pi * self.v_fov / (180 * 2)) / self.height
        self.p_matrix: np.matrix = self.create_proj()
        self.lookat: None | Pose = None

    def create_proj(self) -> np.matrix:
        return np.matrix([[self.f / self.sx, 0, self.width / 2, 0],
                          [0, self.f / self.sy, self.height / 2, 0],
                          [0, 0, 1, 0]])

    def to_camera_space_matrix(self, v_in_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outliers = np.concat((np.asarray(v_in_cam[2, :] < self.near).nonzero()[1],
                              np.asarray(v_in_cam[2, :] > self.far).nonzero()[1]))
        v_in_cam[:, outliers] = np.asarray([0, 0, 0, 1])[:, None]
        return self.p_matrix @ v_in_cam, outliers

    def look_at(self, target: Pose) -> None:
        p1_matrix, p2_matrix = np.asarray(self.pose.to_matrix()), np.asarray(target.to_matrix())
        #Orient Z camera to target
        z_v = np.asarray([0, 0, 1, 1])

        z_cam_to_target = p2_matrix[:3, 3] - p1_matrix[:3, 3]
        cam_current_look_dir = (p1_matrix @ z_v.T)[:3] - p1_matrix[:3, 3]

        #keep same Y camera align with Z target
        z_target = (p2_matrix @ z_v.T)[:3] - p2_matrix[:3, 3]
        y_cam = (p1_matrix @ np.asarray([0, -1, 0, 1]).T)[:3] - p1_matrix[:3, 3]

        ret = np.eye(4, 4)
        target_vectors = np.stack((z_cam_to_target, z_target))
        cam_vectors = np.stack((cam_current_look_dir, y_cam))
        rot, _ = R.align_vectors(target_vectors, cam_vectors)
        ret[:3, :3] = rot.as_matrix()
        self.pose *= Pose.from_matrix(ret)

        # Debug
        # rot1, _ = R.align_vectors(z_cam_to_target, cam_current_look_dir)
        # rot2, _ = R.align_vectors(z_target, y_cam)
        # ret[:3, :3] = rot1.as_matrix() @ rot2.as_matrix() @ R.from_euler('Z', 180, degrees=True).as_matrix()
        # rot1_as_euler = rot1.as_euler("XYZ", degrees=True)
        # rot2_as_euler = rot2.as_euler("XYZ", degrees=True)
        # t = ret[:3,:3] @ y_cam
        # t2 = ret[:3, :3] @ cam_current_look_dir


    def move_x(self, step: float = 0.5) -> None:
        super(Camera, self).move_x(step)
        if self.lookat is not None:
            self.look_at(self.lookat)

    def move_z(self, step: float = 0.5) -> None:
        super(Camera, self).move_z(step)
        if self.lookat is not None:
            self.look_at(self.lookat)

    def move_y(self, step: float = 0.5) -> None:
        super(Camera, self).move_y(step)
        if self.lookat is not None:
            self.look_at(self.lookat)
