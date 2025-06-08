from typing import NamedTuple

from numba import jit
import taichi as ti
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

    def __mul__(self, other:'Pose') :
        return Pose.from_matrix(np.asarray(self.to_matrix() @ other.to_matrix()))

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Pose':
        trans = matrix[:3, 3].tolist()
        euler = R.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
        return Pose(trans[0], trans[1], trans[2], float(euler[0]), float(euler[1]), float(euler[2]))

#@jit
@ti.func
def barycentric(a: ti.math.vec2, b: ti.math.vec2, c: ti.math.vec2, p: ti.math.vec2) :
    """Return barycentric coordinate of 'p' for a triangle abc. If triangle is degenerate return negative values"""
    ab = b - a
    ac = c - a
    pa = a - p
    #ab_ac_pa_x = np.array([b[0] - a[0], c[0] - a[0], a[0] - p[0]], dtype=np.float32)
    #ab_ac_pa_y = np.array([b[1] - a[1], c[1] - a[1], a[1] - p[1]], dtype=np.float32)
    n = ti.math.cross(ti.Vector([ab[0], ac[0], pa[0]]), ti.Vector([ab[1], ac[1], pa[1]]))
    ret = ti.Vector([-1.0, -1.0, -1.0])
    if abs(n[2]) >= 1:
        ret = ti.Vector([1-(n[0]+n[1])/n[2], n[0]/n[2], n[1]/n[2]])
    return ret

@jit
def normalize(v3: np.ndarray) -> np.ndarray:
    return v3 / np.sqrt((v3**2).sum())

#@jit
@ti.func
def bbox_3d_to_2d(a: ti.types.vector(3, ti.float32), b: ti.types.vector(3, ti.float32), c: ti.types.vector(3, ti.float32)) -> tuple[tuple[int, int], tuple[int, int]]:
    bbox_u_l = ti.math.vec2(min(a[0], b[0], c[0]),  min(a[1], b[1], c[1]))
    bbox_b_r =  ti.math.vec2(max(a[0], b[0], c[0]), max(a[1], b[1], c[1]))
    return bbox_u_l, bbox_b_r
