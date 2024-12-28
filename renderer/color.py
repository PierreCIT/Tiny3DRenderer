import numpy as np

from enum import Enum

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
        return np.asarray([self.r, self.g, self.b, self.a])

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


class ColorMap(Enum):
    WHITE = Color(255, 255, 255, 255)
    RED = Color(255,0, 0,255)
    GREEN = Color(0,255, 0,255)
    BLUE = Color(0,0, 255,255)
    CYAN = Color(0,255, 255,255)
    MAGENTA = Color(255,0, 255,255)
    YELLOW = Color(255,255, 0,255)

