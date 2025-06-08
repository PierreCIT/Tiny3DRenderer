from collections.abc import Callable
from logging import getLogger

import taichi as ti
import numpy as np

from .camera import Camera
from .color import Color, ColorMap
from .geometry import Pose, barycentric, normalize, bbox_3d_to_2d
from .light import Light
from .model import Model
from .shaders import draw_triangle

log = getLogger(__name__)

AXES_VECTORS = np.array([[0, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         [-1, 0, 0, 1],
                         [0, -1, 0, 1],
                         [0, 0, -1, 1],], dtype=float)

class RenderMode():
    WIREFRAME = "WIREFRAME"
    FACES_NO_LIGHT_RANDOM_COLORS = "FACES_NO_LIGHT_RANDOM_COLORS"
    FACES_NO_MAT = "FACES_NO_MAT"
    FACES = "FACES"


class Renderer:
    def __init__(self, res: tuple[int, int], mode: RenderMode = RenderMode.WIREFRAME):
        self.default_image = np.full((res[1], res[0], 4), [0, 0, 0, 255], dtype=np.uint8)
        self.image: np.ndarray = self.default_image.copy()
        self.zbuffer: np.ndarray = np.full(self.image.shape[:2], np.inf, dtype=np.float32)
        self.render_mode: RenderMode = mode
        self.models: list[Model] | None = None
        self.camera: Camera | None = None
        self.show_world_axes: bool = True
        self.show_objects_axes: bool = True
        self.instances_map: np.ndarray  = np.full(self.image.shape[:2], 0, dtype=np.uint32)
        self.lights: list[Light] = [Light()]
        self._draw_triangle: Callable[[tuple[int, int, float],  tuple[int, int, float], tuple[int, int, float], list[Color], tuple[float, float, float] | None, int | None], None] = self.draw_triangle_jit
        self.final_zbuffer: np.ndarray | None = None
        self.final_image: np.ndarray | None = None
        self.final_instances_map: np.ndarray | None = None

    def draw(self) -> np.ndarray:
        if self.models is not None:
            drawing_func: Callable[[np.ndarray, np.ndarray, np.ndarray, Color, None | int], None] | None = None
            # if self.render_mode.value == RenderMode.WIREFRAME.value:
            #     drawing_func = self.draw_wireframe
            # elif self.render_mode.value == RenderMode.FACES_NO_LIGHT_RANDOM_COLORS.value:
            #     log.debug("Rendering 'No light rnd colors'")
            #     drawing_func = self.draw_faces_no_light_rnd_colors
            # elif self.render_mode.value == RenderMode.FACES_NO_MAT.value:
            #     drawing_func = self.draw_faces_no_mat
            #     log.debug("Rendering 'Faces'")
            # elif self.render_mode.value == RenderMode.FACES.value:
            #     drawing_func = self.draw_faces
            #     log.debug("Rendering 'Faces'")
            drawing_func = self.draw_faces
            log.debug("Rendering 'Faces'")
            if drawing_func is None:
                log.error("No render_mode set")
            else:
                for idx, m in enumerate(self.models):
                    if m.visible:
                        v = m.get_vertices()
                        c = m.diffuse_mat if m.diffuse_mat is not None else Color()
                        drawing_func(self.project_to_camera(self.camera, v), v, m.get_faces(), Color(), idx+1)
                        if self.show_objects_axes:
                            self.draw_object_axes(m.pose)
        self.draw_world_axes()
        self.final_image, self.final_instances_map, self.final_zbuffer = self.image, self.instances_map, self.zbuffer
        return self.final_image #np.flipud(self.image)

    def clear(self):
        self.image = self.default_image.copy()
        self.zbuffer = np.full(self.image.shape[:2], np.inf, dtype=np.float32)
        self.instances_map = np.full(self.image.shape[:2], 0, dtype=np.uint32)

    def set_models(self, models: list[Model]) -> None:
        self.models = models

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera
        # TODO: maybe update buffer image or have the Camera have its own buffer

    def project_to_camera(self, cam: Camera, vectors: np.ndarray) -> np.ndarray:
        v_in_cam = cam.pose.to_matrix().I @ vectors.T
        p = cam.to_camera_space_matrix(v_in_cam)
        projected, outliers = np.asarray(p[0].T), p[1]
        projected[outliers] = [-1, -1, 1]
        projected = projected * (1 / projected[:, 2])[:, None]
        projected[:, 2] = np.asarray(v_in_cam.T[:, 2]).squeeze()
        return projected

    def draw_point(self, p: tuple[int, int], color: Color | np.ndarray, intensity: float = 1.0) -> None:
        if -1 < p[1] < self.image.shape[0] and -1 < p[0] < self.image.shape[1]:
            if not isinstance(color, np.ndarray):
                color = color.to_array()
            color[:3] = color[:3] * intensity
            self.image[p[1], p[0]] = color.astype(self.image.dtype)

    def draw_line(self, a: tuple[int, int], b: tuple[int, int], color: Color) -> None:
        steep = False
        x0, y0, x1, y1 = a[0], a[1], b[0], b[1]
        if abs(x0 - x1) < abs(y0 - y1):
            steep = True
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        derror2 = abs(dy) * 2
        error2 = 0
        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                self.draw_point((y, x), color)
            else:
                self.draw_point((x, y), color)
            error2 += derror2
            if error2 > dx:
                y += 1 if dy > 0 else -1
                error2 -= dx * 2

    def _draw_line_same_y(self, y: int, x0: int, x1: int, color: Color) -> None:
        if -1 < y < self.image.shape[0] and -1 < x0 < self.image.shape[1] and -1 < x1 < self.image.shape[1]:
            if x0 > x1: x0, x1 = x1, x0
            self.image[y, x0: x1 + 1] = color.to_array()

    def draw_triangle_old_school(self, a: tuple[int, int, float], b: tuple[int, int, float], c: tuple[int, int, float], colors: list[Color],  f_normal: tuple[float, float, float] | None = None, idx: int | None = None) -> None:
        color = colors[0]
        # if line
        if a == b: self.draw_line(a, c, color); return
        if a == c: self.draw_line(a, b, color); return
        if b == c: self.draw_line(a, b, color); return
        # Sort a, b, c from lower to higher
        if a[1] > b[1]: a, b = b, a
        if a[1] > c[1]: a, c = c, a
        if b[1] > c[1]: b, c = c, b
        s_a_to_b_x = (b[0] - a[0]) / (b[1] - a[1]) if b[1] - a[1] != 0 else None
        s_b_to_c_x = (c[0] - b[0]) / (c[1] - b[1]) if c[1] - b[1] != 0 else None
        s_a_to_c_x = (c[0] - a[0]) / (c[1] - a[1])
        a_to_c_origin = a[0] - s_a_to_c_x * a[1]

        if s_a_to_b_x is None:
            self._draw_line_same_y(a[1], a[0], b[0], color)
        else:
            a_to_b_origin = a[0] - s_a_to_b_x * a[1]
            for k in range(b[1] - a[1] + 1):
                y = k + a[1]
                A_to_B_0 = int(a_to_b_origin + s_a_to_b_x * y)
                A_to_C_0 = int(a_to_c_origin + s_a_to_c_x * y)
                self._draw_line_same_y(y, A_to_C_0, A_to_B_0, color)

        if s_b_to_c_x is None:
            self._draw_line_same_y(b[1], b[0], c[0], color)
        else:
            b_to_c_origin = b[0] - s_b_to_c_x * b[1]
            for k in range(c[1] - b[1] + 1):
                y = k + b[1]
                B_to_C_0 = int(b_to_c_origin + s_b_to_c_x * y)
                A_to_C_0 = int(a_to_c_origin + s_a_to_c_x * y)
                self._draw_line_same_y(y, A_to_C_0, B_to_C_0, color)

    def draw_triangle(self, a: tuple[int, int, float], b: tuple[int, int, float], c: tuple[int, int, float], colors: list[Color], f_normal: tuple[float, float, float] | None = None, idx: int | None = None) -> None:
        """Colors for a, b, c respectively"""
        bbox_u_l, bbox_b_r = bbox_3d_to_2d(a, b, c)
        bbox_u_l, bbox_b_r = (max(0, bbox_u_l[0]), max(0, bbox_u_l[1])), (min(self.image.shape[1] - 1, bbox_b_r[0]), min(self.image.shape[0] - 1, bbox_b_r[1]))
        f_n = np.asarray(f_normal) if f_normal is not None else None
        if len(self.lights) > 0 and f_n is not None:
            # TODO: Handle multiple lights
            intensity = np.vdot(f_n, self.lights[0].dir)
        else:
            intensity = None
        a_colors = np.asarray([co.to_array() for co in colors], dtype=np.uint8)
        for i in range(bbox_u_l[0], bbox_b_r[0] + 1):
            for j in range(bbox_u_l[1], bbox_b_r[1] + 1):
                p = (i, j)
                u = barycentric(a[:2], b[:2], c[:2], p)
                if u[0] < 0 or u[1] < 0 or u[2] < 0: continue # Invalid triangle
                z = u[0] * a[2] + u[1] * b[2] + u[2] * c[2]
                if self.zbuffer[j, i] < z: continue
                self.zbuffer[j, i] = z
                if idx is not None:  self.instances_map[j, i] = idx
                if intensity is not None:
                    if intensity > 0:
                        self.draw_point(p, Color.interpolate_numpy(a_colors, u), intensity)
                    else:
                        self.draw_point(p, Color(z, 0, 0))
                else:
                    self.draw_point(p, Color.interpolate_numpy(a_colors, u))

    def draw_triangle_jit(self, a: tuple[int, int, float], b: tuple[int, int, float], c: tuple[int, int, float], colors: list[Color], f_normal: tuple[float, float, float] | None = None, idx: int | None = None) -> None:
        lights_dir = np.asarray([l.dir for l in self.lights], dtype=np.float32)
        lights_color = np.asarray([l.color.to_array() for l in self.lights], dtype=np.uint8)
        colors = np.asarray([co.to_array() for co in colors], dtype=np.uint8)

        l_dir = ti.Vector(lights_dir[0])
        f_n = ti.Vector(f_normal)
        draw_triangle(self.image,
                                                                     self.zbuffer,
                                                                     self.instances_map,
                                                                     l_dir,
                                                                     lights_color,
                                                                     ti.math.vec3(a[0], a[1], a[2]),
                                                                     ti.math.vec3(b[0], b[1], b[2]),
                                                                     ti.math.vec3(c[0], c[1], c[2]),
                                                                     colors,
                                                                     f_n,
                                                                     idx)


    def draw_object_axes(self, o_p: Pose) -> None:
        axes_in_object = AXES_VECTORS[:4].copy()
        axes_in_object[:, :3] /= 4
        axes_in_object = (o_p.to_matrix() @ axes_in_object.T).T
        p = np.round(self.project_to_camera(self.camera, axes_in_object), decimals=0).astype(np.int32)
        origin, x, y, z = p[0], p[1], p[2], p[3]
        self.draw_line((origin[0], origin[1]), (x[0], x[1]), ColorMap.RED.value)
        self.draw_line((origin[0], origin[1]), (y[0], y[1]), ColorMap.GREEN.value)
        self.draw_line((origin[0], origin[1]), (z[0], z[1]), ColorMap.BLUE.value)


    def draw_world_axes(self):
        axis_scale = 4
        a = (AXES_VECTORS * axis_scale)
        a[:, 3] /= axis_scale
        p = np.round(self.project_to_camera(self.camera, a), decimals=0).astype(np.int32)
        origin, x, y, z, m_x, m_y, m_z = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        self.draw_line((origin[0], origin[1]), (x[0], x[1]), ColorMap.RED.value)
        self.draw_line((origin[0], origin[1]), (y[0], y[1]), ColorMap.GREEN.value)
        self.draw_line((origin[0], origin[1]), (z[0], z[1]), ColorMap.BLUE.value)

        self.draw_line((origin[0], origin[1]), (m_x[0], m_x[1]), ColorMap.MAGENTA.value)
        self.draw_line((origin[0], origin[1]), (m_y[0], m_y[1]), ColorMap.YELLOW.value)
        self.draw_line((origin[0], origin[1]), (m_z[0], m_z[1]), ColorMap.CYAN.value)


    def draw_wireframe(self, v_screen: np.ndarray, v: np.ndarray, f: np.ndarray, c: Color = Color(), idx: int | None = None) -> None:
        c_a =  c.to_array()
        for f_i in range(f.shape[0]):
            for k in range(3):
                v0 = np.round(v_screen[f[f_i][k]], decimals=0).astype(np.int32)
                v1 = np.round(v_screen[f[f_i][(k + 1) % 3]], decimals=0).astype(np.int32)
                if not (v0[0] < 0 and v1[0] < 0):
                    self.draw_line((v0[0], v0[1]), (v1[0], v1[1]), color=c_a)


    def draw_faces_no_light(self, v_screen: np.ndarray, v: np.ndarray, f: np.ndarray, c: Color = Color(), idx: int | None = None) -> None:
        for f_i in range(f.shape[0]):
            v0 = np.round(v_screen[f[f_i][0]], decimals=0).astype(np.int32)
            v1 = np.round(v_screen[f[f_i][1]], decimals=0).astype(np.int32)
            v2 = np.round(v_screen[f[f_i][2]], decimals=0).astype(np.int32)
            if not (v0[0] < 0 and v1[0] < 0):
                self._draw_triangle((v0[0], v0[1], v[f[f_i][0]][0, 2]),
                                    (v1[0], v1[1], v[f[f_i][1]][0, 2]),
                                    (v2[0], v2[1], v[f[f_i][2]][0, 2]),
                                    colors=[c],
                                    f_normal=None)


    def draw_faces_no_light_rnd_colors(self, v_screen: np.ndarray, v: np.ndarray, f: np.ndarray, c: Color = None, idx: int | None = None) -> None:
        for f_i in range(f.shape[0]):
            v0 = np.round(v_screen[f[f_i][0]], decimals=0).astype(np.int32)
            v1 = np.round(v_screen[f[f_i][1]], decimals=0).astype(np.int32)
            v2 = np.round(v_screen[f[f_i][2]], decimals=0).astype(np.int32)
            if not (v0[0] < 0 and v1[0] < 0):
                self._draw_triangle((v0[0], v0[1], v[f[f_i][0]][0, 2]),
                                    (v1[0], v1[1], v[f[f_i][1]][0, 2]),
                                    (v2[0], v2[1], v[f[f_i][2]][0, 2]),
                                    colors=[Color.random()],
                                    f_normal=None)

    def draw_faces_no_mat(self, v_screen: np.ndarray, v: np.ndarray, f: np.ndarray, c: Color = None, idx: int | None = None) -> None:
        for f_i in range(f.shape[0]):
            v0 = np.round(v_screen[f[f_i][0]][:2], decimals=0).astype(np.int32)
            v1 = np.round(v_screen[f[f_i][1]][:2], decimals=0).astype(np.int32)
            v2 = np.round(v_screen[f[f_i][2]][:2], decimals=0).astype(np.int32)
            f_normal = normalize(np.cross((v[f[f_i][1]] - v[f[f_i][0]])[:, :3], (v[f[f_i][2]] - v[f[f_i][0]])[:, :3])[0])
            self._draw_triangle((v0[0], v0[1], v_screen[f[f_i][0]][2]),
                                    (v1[0], v1[1], v_screen[f[f_i][1]][2]),
                                    (v2[0], v2[1], v_screen[f[f_i][2]][2]),
                                    colors=[ColorMap.WHITE.value, ColorMap.WHITE.value, ColorMap.WHITE.value],
                                    f_normal=f_normal, idx=idx)

    def draw_faces(self, v_screen: np.ndarray, v: np.ndarray, f: np.ndarray, c: Color = None, idx: int | None = None) -> None:
        for f_i in range(f.shape[0]):
            v0 = np.round(v_screen[f[f_i][0]][:2], decimals=0).astype(np.int32)
            v1 = np.round(v_screen[f[f_i][1]][:2], decimals=0).astype(np.int32)
            v2 = np.round(v_screen[f[f_i][2]][:2], decimals=0).astype(np.int32)
            f_normal = normalize(np.cross((v[f[f_i][1]] - v[f[f_i][0]])[:, :3], (v[f[f_i][2]] - v[f[f_i][0]])[:, :3])[0])
            self._draw_triangle((v0[0], v0[1], v_screen[f[f_i][0]][2]),
                                    (v1[0], v1[1], v_screen[f[f_i][1]][2]),
                                    (v2[0], v2[1], v_screen[f[f_i][2]][2]),
                                    colors=[ColorMap.WHITE.value, ColorMap.WHITE.value, ColorMap.WHITE.value],
                                    f_normal=f_normal, idx=idx)
