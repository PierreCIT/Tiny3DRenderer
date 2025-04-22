from enum import Enum

import numpy as np
from numba import jit

@jit
def interpolate_numpy(c_array: np.ndarray, w: np.ndarray, ignore_alpha: bool = True) -> np.ndarray:
    """Interpolate colors based on weights (must be same size)"""
    n_w = w / w.sum()
    r = (n_w * c_array.T).T.sum(axis=0)
    if ignore_alpha:
        r[3] = 255
    return np.asarray([r[0], r[1], r[2], r[3]], dtype=np.uint8)

class Color:
    """RGBA colors in [0, 255]"""
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, a: int = 255):
        self.r: int = r
        self.g: int = g
        self.b: int = b
        self.a: int = a

    def __eq__(self, other: 'Color'):
        return (self.to_array() == other.to_array()).sum() == 4

    def to_array(self) -> np.ndarray:
        return np.asarray([self.r, self.g, self.b, self.a], dtype=np.uint8)

    @staticmethod
    def random() -> 'Color':
        """Return a random color without transparency"""
        return Color(*np.random.randint(0, 255, 3).tolist())

    @staticmethod
    def interpolate(colors: list['Color'], w: np.ndarray, ignore_alpha: bool = True) -> 'Color':
        """Interpolate colors based on weights (must be same size)"""
        n_w = w / w.sum()
        c_array = np.asarray([c.to_array() for c in colors])
        r = (n_w * c_array.T).T.sum(axis=0)
        return Color(round(r[0]), round(r[1]), round(r[2]), 255 if ignore_alpha else round(r[3]))

    @staticmethod
    def interpolate_numpy(c_array: np.ndarray, w: np.ndarray, ignore_alpha: bool = True) -> np.ndarray:
        return interpolate_numpy(c_array, w, ignore_alpha)

    @staticmethod
    def interpolate_conv_to_numpy(colors: list['Color'], w: np.ndarray, ignore_alpha: bool = True) -> np.ndarray:
        return Color.interpolate_numpy(np.asarray([c.to_array() for c in colors], dtype=np.uint8), w, ignore_alpha)

class ColorMap(Enum):
    WHITE = Color(255, 255, 255, 255)
    RED = Color(255,0, 0,255)
    GREEN = Color(0,255, 0,255)
    BLUE = Color(0,0, 255,255)
    CYAN = Color(0,255, 255,255)
    MAGENTA = Color(255,0, 255,255)
    YELLOW = Color(255,255, 0,255)

