import json
from logging import getLogger
from pathlib import Path

from .camera import Camera
from .model import Model
from .renderer import RenderMode
from .scene import Scene

log = getLogger(__name__)


class Config:

    def __init__(self, path: Path):
        self.path: Path = path
        self.scenes: dict[str, Scene] = dict()
        self.scenes_conf: dict[str, dict] = dict()
        self.models: dict[str, Model] = dict()
        self.loaded_scene: str | None = None
        self.default_render_mode: RenderMode = RenderMode.WIREFRAME
        self._init()

    def _init(self) -> None:
        with open(self.path, 'r') as f:
            c = json.load(f)
        self.models = {name: Model((Path(__file__).parent / m).resolve(), name)
                       for name, m in c["models"].items()}
        for scene in c["scenes"]:
            for k, s in scene.items():
                self.scenes_conf[k] = s

    def load(self, scene_name: str) -> Scene:
        if scene_name not in self.scenes_conf.keys():
            raise KeyError(f"Scene '{scene_name}' not found.")
        if scene_name != self.loaded_scene:
            #TODO: unload models no longer used
            objects3d = Scene.read_scene_objects(self.scenes_conf[scene_name]["objects"])
            models: dict[str, Model] = dict()
            for name, o in objects3d.items():
                self.models[name].set_pose(o.pose)
                models = {name: self.models[name] for name, o in objects3d.items()}

            cameras = Scene.read_scene_cameras(self.scenes_conf[scene_name]["cameras"])
            if len(cameras) == 0:
                cameras = {"default": Camera(name="default")}
            active_camera = self.scenes_conf[scene_name]["active_camera"] if "active_camera" in self.scenes_conf[scene_name] else list(cameras.keys())[0]
            render_mode = "FACES" #RenderMode[self.scenes_conf[scene_name]["render_mode"]] if "render_mode" in self.scenes_conf[scene_name] else self.default_render_mode

            self.scenes[scene_name] = Scene(models, cameras, active_camera, render_mode)
            self.scenes[scene_name].load()
        else:
            log.info(f"Scene '{scene_name}' already loaded. Nothing to do.")
        return self.scenes[scene_name]

    def get_loaded_scene_models(self) -> list[Model]:
        return [self.models[k] for k in self.scenes[self.loaded_scene].get_models_names()]
