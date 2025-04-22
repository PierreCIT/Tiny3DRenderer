import numpy as np
import pytest
from pathlib import Path

from renderer.camera import Camera
from renderer.color import Color, ColorMap
from renderer.geometry import Pose, normalize
from renderer.light import Light
from renderer.model import Model
from renderer.renderer import Renderer, RenderMode
from renderer.tool import write_numpy_to_png

TEST_DIR = Path(__file__).parent / "output/"
TEST_DIR.mkdir(parents=True, exist_ok=True)
AFRICAN_HEAD_OBJ_PATH = Path(__file__).parent / "../../obj/african_head/african_head.obj"


@pytest.fixture
def renderer_wireframe() -> Renderer:
    return Renderer((20, 20), RenderMode.WIREFRAME)


@pytest.fixture
def cam_renderer_wireframe() -> tuple[Camera, Renderer]:
    cam = Camera("TestCam", Pose(2, -2, 2, 0, 0, 45))
    renderer = Renderer(cam.res, RenderMode.WIREFRAME)
    renderer.set_camera(cam)
    return cam, renderer


@pytest.fixture
def cam_renderer_face_rnd_colors() -> tuple[Camera, Renderer]:
    cam = Camera("TestCam", Pose(2, -2, 2, 0, 0, 45))
    renderer = Renderer(cam.res, RenderMode.NO_LIGHT_RANDOM_COLORS)
    renderer.set_camera(cam)
    return cam, renderer


def test_draw_point(renderer_wireframe: Renderer) -> None:
    c = Color()
    renderer_wireframe.draw_point((10, 10), c)
    n_0 = np.where(renderer_wireframe.image != [0, 0, 0, 255])
    assert len(n_0[0]) == 3
    for i in range(len(n_0)):
        assert renderer_wireframe.image[n_0[0][i], n_0[1][i], n_0[2][i]] == c.to_array()[i]
    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_point_test.png")


def test_draw_line_visual(renderer_wireframe: Renderer) -> None:
    r = ColorMap.RED.value
    renderer_wireframe.draw_line((10, 0), (11, 19), r)
    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_line_test.png")


def test_draw_line_diagonal_visual(renderer_wireframe: Renderer) -> None:
    r = ColorMap.RED.value
    renderer_wireframe.draw_line((0, 0), (19, 19), r)
    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_line_test_diagonal.png")


def test_draw_line_almost_vertical_visual(renderer_wireframe: Renderer) -> None:
    r = ColorMap.RED.value
    renderer_wireframe.draw_line((19, 0), (0, 10), r)
    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_line_test_almost_vertical.png")


def test_draw_lines_visual(renderer_wireframe: Renderer) -> None:
    im_size = renderer_wireframe.image.shape
    r = ColorMap.RED.value
    b = ColorMap.BLUE.value
    g = ColorMap.GREEN.value
    renderer_wireframe.draw_line((0, 0), (int(round(im_size[0] / 2, 0)), int(round(im_size[1] / 2, 0))), r)
    renderer_wireframe.draw_line((int(round(im_size[0] / 2, 0)), int(round(im_size[1] / 2, 0))), (im_size[0] - 1, im_size[1] - 1),
                                 b)
    renderer_wireframe.draw_line((0, im_size[1] - 1), (im_size[0] - 1, 0), g)
    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_lines_test.png")


def test_draw_rectangles_visual(renderer_wireframe: Renderer) -> None:
    im_size = renderer_wireframe.image.shape
    r = ColorMap.RED.value
    b = ColorMap.BLUE.value
    g = ColorMap.GREEN.value

    renderer_wireframe.draw_line((0, 0), (0, im_size[1] - 1), r)
    renderer_wireframe.draw_line((0, 0), (im_size[0] - 1, 0), r)
    renderer_wireframe.draw_line((im_size[0] - 1, 0), (im_size[0] - 1, im_size[1] - 1), r)
    renderer_wireframe.draw_line((0, im_size[1] - 1), (im_size[0] - 1, im_size[1] - 1), r)

    renderer_wireframe.draw_line((1, 1), (1, im_size[1] - 2), b)
    renderer_wireframe.draw_line((1, 1), (im_size[0] - 2, 1), b)
    renderer_wireframe.draw_line((im_size[0] - 2, 1), (im_size[0] - 2, im_size[1] - 2), b)
    renderer_wireframe.draw_line((1, im_size[1] - 2), (im_size[0] - 2, im_size[1] - 2), b)

    renderer_wireframe.draw_line((2, 2), (2, im_size[1] - 3), g)
    renderer_wireframe.draw_line((2, 2), (im_size[0] - 3, 2), g)
    renderer_wireframe.draw_line((im_size[0] - 3, 2), (im_size[0] - 3, im_size[1] - 3), g)
    renderer_wireframe.draw_line((2, im_size[1] - 3), (im_size[0] - 3, im_size[1] - 3), g)

    write_numpy_to_png(renderer_wireframe.image, TEST_DIR / "draw_rectangles_test.png")


def test_draw_world_axes(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    renderer.draw_world_axes()
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_world_axes_test.png")


def test_draw_world_axes_camera_looking_at_origin() -> None:
    init_cam_pose = Pose(1.5, -1, 0.5, 0, 0, 0)
    cam = Camera(name="default", pose=init_cam_pose)
    cam.look_at(Pose())
    r = Renderer(cam.res, RenderMode.WIREFRAME)
    r.set_camera(cam)
    write_numpy_to_png(r.draw(), TEST_DIR / "draw_world_axes_camera_looking_at_origin_test.png")


def test_draw_triangle_old_school(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    t1 = {"a": (19, 19), "b": (30, 30), "c": (30, 19), "colors": [ColorMap.RED.value]}
    line = {"a": (20, 20), "b": (20, 20), "c": (30, 30), "colors": [ColorMap.MAGENTA.value]}
    renderer.draw_triangle_old_school(**t1)
    renderer.draw_triangle_old_school(**line)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangle_old_school_and_line_test.png")


def test_draw_triangles_old_school(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    t1 = {"a": (10, 70), "b": (50, 160), "c": (70, 80), "colors": [ColorMap.RED.value]}
    t2 = {"a": (180, 50), "b": (150, 1), "c": (70, 180), "colors": [ColorMap.WHITE.value]}
    t3 = {"a": (180, 150), "b": (120, 160), "c": (130, 180), "colors": [ColorMap.GREEN.value]}
    line = {"a": (20, 20), "b": (20, 20), "c": (30, 32), "colors": [ColorMap.MAGENTA.value]}
    renderer.draw_triangle_old_school(**t1)
    renderer.draw_triangle_old_school(**t2)
    renderer.draw_triangle_old_school(**t3)
    renderer.draw_triangle_old_school(**line)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangles_old_school_test.png")


def test_draw_triangles(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    t1 = {"a": (10, 70, 0), "b": (50, 160, 0), "c": (70, 80, 0), "colors": [ColorMap.RED.value, ColorMap.RED.value, ColorMap.RED.value]}
    t2 = {"a": (180, 50, 0), "b": (150, 1, 0), "c": (70, 180, 0), "colors": [ColorMap.WHITE.value, ColorMap.WHITE.value, ColorMap.WHITE.value]}
    t3 = {"a": (180, 150, 0), "b": (120, 160, 0), "c": (130, 180, 0), "colors": [ColorMap.GREEN.value, ColorMap.GREEN.value, ColorMap.GREEN.value]}
    line = {"a": (20, 20, 0), "b": (20, 20, 0), "c": (30, 32, 0), "colors": [ColorMap.MAGENTA.value, ColorMap.MAGENTA.value, ColorMap.MAGENTA.value]}
    renderer.draw_triangle(**t1)
    renderer.draw_triangle(**t2)
    renderer.draw_triangle(**t3)
    renderer.draw_triangle(**line)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangles_test.png")


def test_draw_triangles_interpolate(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    t1 = {"a": (10, 10, 0), "b": (100, 30, 0), "c": (190, 160, 0), "colors": [ColorMap.RED.value, ColorMap.GREEN.value, ColorMap.BLUE.value]}
    t2 = {"a": (200, 270, 0), "b": (250, 360, 0), "c": (270, 280, 0), "colors": [ColorMap.RED.value, ColorMap.GREEN.value, ColorMap.BLUE.value]}
    renderer.draw_triangle(**t1)
    renderer.draw_triangle(**t2)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangles_interpolate_test.png")


def test_draw_triangle_zbuffer(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    t1 = {"a": (10, 10, 0), "b": (10, 300, 0), "c": (300, 150, 2), "colors": [ColorMap.RED.value, ColorMap.RED.value, ColorMap.BLUE.value]}
    t2 = {"a": (300, 10, 0), "b": (300, 300, 0), "c": (10, 150, 2), "colors": [ColorMap.GREEN.value, ColorMap.GREEN.value, ColorMap.BLUE.value]}
    renderer.draw_triangle(**t1)
    renderer.draw_triangle(**t2)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangle_zbuffer_test.png")
    write_numpy_to_png(renderer.zbuffer, TEST_DIR / "draw_triangle_zbuffer_Z_test.npy")


def test_draw_triangle_zbuffer_light(cam_renderer_wireframe) -> None:
    cam, renderer = cam_renderer_wireframe
    renderer.lights = [Light("TestLight", Pose(), dir=np.asarray([2, 0, 2]))]
    t1 = {"a": (10, 10, 0), "b": (10, 300, 0), "c": (300, 150, 20), "colors": [ColorMap.RED.value, ColorMap.RED.value, ColorMap.RED.value]}
    t2 = {"a": (300, 10, 0), "b": (300, 300, 0), "c": (10, 150, 20), "colors": [ColorMap.RED.value, ColorMap.RED.value, ColorMap.RED.value]}
    normal_t1 = normalize(np.cross(np.asarray(t1["c"]) - t1["a"], np.asarray(t1["b"]) - t1["a"])).tolist()
    normal_t2 = normalize(np.cross(np.asarray(t2["b"]) - t2["a"], np.asarray(t2["c"]) - t2["a"])).tolist()
    renderer.draw_triangle(**t1, f_normal=normal_t1)
    renderer.draw_triangle(**t2, f_normal=normal_t2)
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_triangle_zbuffer_light_test.png")
    write_numpy_to_png(renderer.zbuffer, TEST_DIR / "draw_triangle_zbuffer_Z_light_test.npy")


def test_draw_face_random_colors(cam_renderer_face_rnd_colors) -> None:
    cam, renderer = cam_renderer_face_rnd_colors
    model = Model(AFRICAN_HEAD_OBJ_PATH, "Face", Pose())
    model.load()
    renderer.set_models([model])
    renderer.draw()
    write_numpy_to_png(renderer.image, TEST_DIR / "draw_face_random_colors_test.png")

