from logging import getLogger

from numba import jit, njit, prange
import numpy as np

from .color import interpolate_numpy
from .geometry import barycentric, bbox_3d_to_2d

log = getLogger(__name__)

@jit
def draw_point(image: np.ndarray, p: tuple[int, int], color: np.ndarray, intensity: float = 1.0) -> None:
    if -1 < p[1] < image.shape[0] and -1 < p[0] < image.shape[1]:
        color[:3] = color[:3] * intensity
        image[p[1], p[0]] = color.astype(image.dtype)

@njit(parallel=True)
def draw_triangle(image: np.ndarray,
                  zbuffer: np.ndarray,
                  instances_map: np.ndarray,
                  lights_dir: np.ndarray,
                  lights_color: np.ndarray,
                  a: tuple[int, int, float], b: tuple[int, int, float], c: tuple[int, int, float],
                  colors: np.ndarray,
                  f_normal: tuple[float, float, float] | None = None,
                  idx: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colors for a, b, c respectively"""
    bbox_u_l, bbox_b_r = bbox_3d_to_2d(a, b, c)
    bbox_u_l, bbox_b_r = (max(0, bbox_u_l[0]), max(0, bbox_u_l[1])), (min(image.shape[1] - 1, bbox_b_r[0]),
                                                                      min(image.shape[0] - 1, bbox_b_r[1]))
    f_n = np.asarray(f_normal, dtype=np.float32) if f_normal is not None else None
    if lights_dir.size > 0 and f_n is not None:
        # TODO: Handle multiple lights
        intensity = np.dot(lights_dir, f_n)[0]
    else:
        intensity = None
    for i in prange(bbox_u_l[0], bbox_b_r[0] + 1):
        for j in prange(bbox_u_l[1], bbox_b_r[1] + 1):
            p = (i, j)
            u = barycentric(a[:2], b[:2], c[:2], p)
            if u[0] < 0 or u[1] < 0 or u[2] < 0: continue  # Invalid triangle
            z = u[0] * a[2] + u[1] * b[2] + u[2] * c[2]
            closest_to_camera = z < zbuffer[j, i]
            if not closest_to_camera: continue
            zbuffer[j, i] = z
            if idx is not None:  instances_map[j, i] = idx
            if intensity is not None:
                if intensity > 0:
                    draw_point(image, p, interpolate_numpy(colors, u), intensity)
                else:
                    draw_point(image, p, np.asarray([150, 0, 0, 255], dtype=np.uint8))
            else:
                draw_point(image, p, np.asarray([0, 150, 0, 255], dtype=np.uint8))
    return image, zbuffer, instances_map
