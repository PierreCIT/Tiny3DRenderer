from typing import Any

import numpy as np

from .geometry import Pose
from .model import Model
from .object3d import Object3D
from .camera import Camera
from .renderer import Renderer, RenderMode


class Scene:
    def __init__(self, models: dict[str, Model], cameras: dict[str, Camera], active_camera_name: str, mode: RenderMode = RenderMode.WIREFRAME):
        self.models: dict[str, Model] = models
        self.cameras: dict[str, Camera] = cameras
        self.renderer: Renderer | None = None
        self.renderer_mode: RenderMode =  mode

        self._active_camera: str = active_camera_name

        self.set_active_camera(active_camera_name)

    def set_active_camera(self, cam_name: str) -> None:
        self.cameras[self._active_camera].visible = True
        self._active_camera = cam_name
        self.cameras[self._active_camera].visible = False

    def get_active_camera_name(self) -> str:
        return self._active_camera

    @staticmethod
    def read_scene_objects(objects: list[dict[str, Any]]) -> dict[str, Object3D]:
        ret = {}
        for dict_o in objects:
            o = Object3D(**dict_o)
            ret[o.name] = o
        return ret

    @staticmethod
    def read_scene_cameras(cameras: list[dict[str, Any]]) -> dict[str, Camera]:
        ret = {}
        for dict_c in cameras:
            c = Camera(**dict_c)
            ret[c.name] = c
        return ret

    def load(self) -> None:
        [self.models[k].load() for k in self.models.keys()]
        [self.cameras[k].load() for k in self.cameras.keys()]
        self.renderer = Renderer(self.cameras[self._active_camera].res, self.renderer_mode)
        self.renderer.set_models([v for _, v in self.models.items()] + [v for _, v in self.cameras.items()])
        self.renderer.set_camera(self.cameras[self._active_camera])

    #TODO: create unload method

    def get_models_names(self) -> list[str]:
        return list(self.models.keys())

    def set_camera_to_look_at(self, p: Pose | None, cam_name: str | None = None) -> None:
        if cam_name is not None:
            self.cameras[cam_name].lookat = p
        else:
            for k, c in self.cameras.items():
                c.lootak = p

    def draw(self) -> np.ndarray:
        self.renderer.clear()
        img = self.renderer.draw()
        return img

    def get_current_render(self) -> np.ndarray:
        return self.renderer.final_image
