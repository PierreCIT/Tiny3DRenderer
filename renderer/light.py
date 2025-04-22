import numpy as np

from .object3d import Object3D
from .color import Color, ColorMap
from .geometry import Pose, normalize

class Light(Object3D):

    def __init__(self, name: str="default_light", pose: str | Pose = Pose(), dir: np.ndarray=np.asarray([0,1,0]), color: Color=ColorMap.WHITE.value):
        super().__init__(name, pose)
        self.color: Color = color
        self.dir: np.ndarray = normalize(dir)
