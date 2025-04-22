import numpy as np
from logging import getLogger
from pathlib import Path

from PIL import Image

from .object3d import Object3D
from .geometry import Pose

log = getLogger(__name__)


class Model(Object3D):
    def __init__(self, path: Path, name: str, pose: str | Pose = Pose()):
        super().__init__(name, pose)
        # Model data
        self.path: Path = path
        self.V: np.ndarray | None = None
        self.T: np.ndarray | None = None
        self.Vi: np.ndarray | None = None
        self.Ti: np.ndarray | None = None
        self.diffuse_mat: np.ndarray | None = None

        # Renderer options
        self.visible: bool = True

        # Depends on model type
        self.obj_path: Path | None = None
        self.mat_path: Path | None = None
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        found_obj = False
        if self.path.is_dir():
            for f in self.path.rglob("*.obj"):
                if f.suffix == ".obj":
                    found_obj = True
                    self.obj_path = f
            for f in self.path.rglob("*diffuse*.tga"):
                if f.suffix == ".tga":
                    self.mat_path = f
                #TODO: Check for additional materials, texture
        else:
            if self.path.suffix == ".obj":
                self.obj_path = self.path
                found_obj = True
        if not found_obj:
            raise RuntimeError(f"Could not find '.obj' file for model "
                               f"with path: {self.path}")

    def set_pose(self, p: Pose) -> None:
        """Pose is assumed to be the referential in which
        `get_vertices` will express the vertices"""
        self.pose = p

    def load(self) -> None:
        V, T, Vi, Ti = [], [], [], []
        with open(self.obj_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    V.append(np.asarray([float(x) for x in values[1:4]] + [1], dtype=np.float16))
                elif values[0] == 'vt':
                    T.append([float(x) for x in values[1:3]])
                elif values[0] == 'f':
                    Vi.append([int(indices.split('/')[0]) for indices in values[1:]])
                    Ti.append([int(indices.split('/')[1]) for indices in values[1:]])
        self.V, self.T, self.Vi, self.Ti = np.array(V), np.array(T), np.array(Vi) - 1, np.array(Ti) - 1

        if self.mat_path is not None:
            self.diffuse_mat = np.asarray(Image.open(self.mat_path))

    def unload(self) -> None:
        self.V = None
        self.T = None
        self.Vi = None
        self.Ti = None

        self.diffuse_mat = None

    def get_vertices(self) -> np.ndarray:
        return (self.pose.to_matrix() @ self.V.T).T

    def get_vertices_raw(self):
        return self.V

    def get_faces(self) -> np.ndarray:
        return self.Vi
