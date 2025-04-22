import numpy as np
import pytest

from pathlib import Path
from renderer.camera import Camera
from renderer.color import Color, ColorMap
from renderer.geometry import Pose
from renderer.renderer import Renderer, RenderMode
from renderer.tool import write_numpy_to_png

TEST_DIR = Path(__file__).parent / "output/"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def test_pose_from_matrix() -> None:
    A = Pose()
    B = Pose.from_matrix(A.to_matrix())
    assert A == B
    A = Pose(-4, 3, 4, 10, 20, 30)
    B = Pose.from_matrix(A.to_matrix())
    assert A == B



