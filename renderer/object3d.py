from logging import getLogger

import numpy as np

from .geometry import Pose

log = getLogger(__name__)

class Object3D:
    def __init__(self, name: str, pose: str | Pose = Pose()):
        self.name = name
        self.show_axes = True

        # Physics
        self.pose:Pose = Pose(*eval(pose, {}, {pose: 'pose'})) if isinstance(pose, str) else pose
        self.speed: np.ndarray = np.asarray([0, 0, 0])
        self.r_speed: np.ndarray = np.asarray([0, 0, 0])
        self.speed_m: np.ndarray = np.asarray([0, 0, 0]) # momentum
        self.r_speed_m: np.ndarray = np.asarray([0, 0, 0]) # momentum

    def move_z(self, step: float = 0.5) -> None:
        step = Pose(0, 0, step, 0, 0, 0)
        self.pose *= step

    def move_x(self, step: float = 0.5) -> None:
        step = Pose(step, 0, 0, 0, 0, 0)
        self.pose *= step

    def move_y(self, step: float = 0.5) -> None:
        step = Pose(0, step, 0, 0, 0, 0)
        self.pose *= step

    def rotate_z(self, step: float = 10) -> None:
        step = Pose(0, 0, 0, 0, 0, step)
        self.pose *= step

    def rotate_x(self, step: float = 10) -> None:
        step = Pose(0, 0, 0, step, 0, 0)
        self.pose *= step

    def rotate_y(self, step: float = 10) -> None:
        step = Pose(0, 0, 0, 0, step, 0)
        self.pose *= step
