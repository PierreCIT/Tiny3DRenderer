from logging import getLogger

from .geometry import Pose

log = getLogger(__name__)

class Object3D:
    def __init__(self, name: str, pose: str | Pose = Pose()):
        self.name = name
        self.pose = Pose(*eval(pose, {}, {pose: 'pose'})) if isinstance(pose, str) else pose
        self.show_axes = True

    def move_z(self, step: float = 0.5) -> None:
        step = Pose(0, 0, step, 0, 0, 0)
        self.pose *= step

    def move_x(self, step: float = 0.5) -> None:
        step = Pose(step, 0, 0, 0, 0, 0)
        self.pose *= step
