from typing import NamedTuple, Self

from numba import jit
import numpy as np
from scipy.spatial.transform import Rotation as R


class Point2(NamedTuple):
    x: float
    y: float

class Point3(NamedTuple):
    x: float
    y: float
    z: float

class Pose(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_matrix(self) -> np.matrix:
        ret = np.eye(4, 4, dtype=np.float32)
        ret[:3, :3] = R.from_euler("XYZ", [self.roll, self.pitch, self.yaw], degrees=True).as_matrix()
        ret[:3, 3] = [self.x, self.y, self.z]
        return np.matrix(ret)

    def __eq__(self, other:'Pose') -> bool:
        equals = True
        for k in range(6):
            if other[k] - self[k] > 1e-7:
                equals = False
                break
        return equals

    def __mul__(self, other:'Pose') -> Self:
        return Pose.from_matrix(np.asarray(self.to_matrix() @ other.to_matrix()))

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Pose':
        trans = matrix[:3, 3].tolist()
        euler = R.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
        return Pose(trans[0], trans[1], trans[2], float(euler[0]), float(euler[1]), float(euler[2]))

@jit
def barycentric(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int], p: tuple[int, int]) -> np.ndarray:
    """Return barycentric coordinate of 'p' for a triangle abc. If triangle is degenerate return negative values"""
    ab_ac_pa_x = np.array([b[0] - a[0], c[0] - a[0], a[0] - p[0]], dtype=np.float32)
    ab_ac_pa_y = np.array([b[1] - a[1], c[1] - a[1], a[1] - p[1]], dtype=np.float32)
    n = np.cross(ab_ac_pa_x, ab_ac_pa_y)
    if abs(n[2]) >= 1:
        return np.asarray([1-(n[0]+n[1])/n[2], n[0]/n[2], n[1]/n[2]])
    else:
        return np.asarray([-1.0, -1.0, -1.0])
@jit
def normalize(v3: np.ndarray) -> np.ndarray:
    return v3 / np.sqrt((v3**2).sum())

@jit
def bbox_3d_to_2d(a: tuple[int, int, float], b: tuple[int, int, float], c: tuple[int, int, float]) -> tuple[tuple[int, int], tuple[int, int]]:
    bbox_u_l = min(a[0], b[0], c[0]),  min(a[1], b[1], c[1])
    bbox_b_r =  max(a[0], b[0], c[0]), max(a[1], b[1], c[1])
    return bbox_u_l, bbox_b_r
