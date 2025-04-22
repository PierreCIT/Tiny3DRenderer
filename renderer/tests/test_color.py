import numpy as np
import pytest
from pathlib import Path


from renderer.color import Color, ColorMap

TEST_DIR = Path(__file__).parent / "output/"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def test_interpolate() -> None:
    c = Color.interpolate([ColorMap.RED.value, ColorMap.GREEN.value, ColorMap.BLUE.value], np.asarray([0.5, 0.5, 0.5]))
    assert c == Color(85, 85, 85, 255)

def test_interpolate_no_ignore_alpha() -> None:
    c = Color.interpolate([ColorMap.RED.value, ColorMap.GREEN.value, ColorMap.BLUE.value], np.asarray([0.5, 0.5, 0.5]), ignore_alpha=False)
    assert c == Color(85, 85, 85, 255)