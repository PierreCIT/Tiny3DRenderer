from logging import getLogger
from typing import Any

from numba import jit, njit, prange
import taichi as ti
import numpy as np

from .color import interpolate_numpy
from .geometry import barycentric, bbox_3d_to_2d

log = getLogger(__name__)

#@jit
@ti.func
def draw_point(image: ti.types.ndarray(dtype=ti.uint8, ndim=3), p: ti.math.vec2, color: ti.math.vec4, intensity = 1.0) -> None:
    if -1 < p[1] < image.shape[0] and -1 < p[0] < image.shape[1]:
        color[:3] = color[:3] * intensity
        image[int(p[1]), int(p[0]), 0] = color[0]
        image[int(p[1]), int(p[0]), 1] = color[1]
        image[int(p[1]), int(p[0]), 2] = color[2]

#@njit(parallel=True)
@ti.kernel
def draw_triangle(image: ti.types.ndarray(dtype=ti.uint8, ndim=3),
                  zbuffer: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  instances_map: ti.types.ndarray(dtype=ti.uint32, ndim=2),
                  lights_dir: ti.math.vec2,
                  lights_color: ti.types.ndarray(dtype=ti.uint8, ndim=2),
                  a: ti.types.vector(3, ti.f32),
                  b: ti.types.vector(3, ti.f32),
                  c: ti.types.vector(3, ti.f32),
                  colors: ti.types.ndarray(dtype=ti.u8, ndim=2),
                  f_normal: ti.math.vec2,
                  idx: int):
    """Colors for a, b, c respectively"""
    bbox_u_l, bbox_b_r = bbox_3d_to_2d(a, b, c)
    bbox_u_l, bbox_b_r = (max(0, bbox_u_l[0]), max(0, bbox_u_l[1])), (min(image.shape[1] - 1, bbox_b_r[0]),
                                                                      min(image.shape[0] - 1, bbox_b_r[1]))
    intensity = lights_dir.dot(f_normal)
    for i in range(bbox_u_l[0], bbox_b_r[0] + 1):
        for j in range(bbox_u_l[1], bbox_b_r[1] + 1):
            p =  ti.math.vec2(i, j)
            u = barycentric(ti.math.vec2(a[0], a[1]), ti.math.vec2(b[0], b[1]), ti.math.vec2(c[0], c[1]), p)
            if u[0] < 0 or u[1] < 0 or u[2] < 0: continue  # Invalid triangle
            z = u[0] * a[2] + u[1] * b[2] + u[2] * c[2]
            closest_to_camera = z < zbuffer[j, i]
            if not closest_to_camera: continue
            zbuffer[j, i] = z
            instances_map[j, i] = idx
            if intensity > 0:
                draw_point(image, p, interpolate_numpy(colors, u), intensity)
            else:
                draw_point(image, p, ti.Vector([150, 0, 0, 255]))

