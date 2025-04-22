from threading import Thread
from time import monotonic_ns

import numpy as np

from .object3d import Object3D

class Engine:

    def __init__(self, objects: dict[str, Object3D] | None = None) -> None:
        self._objects: dict[str, Object3D] | None = objects
        self._last_loop_t: int = monotonic_ns()
        self._loop: Thread = Thread(target=self._run)
        self._running: bool = False

    def start(self) -> None:
        self._running = True
        self._loop.start()

    def stop(self, timeout_sec: float | None = 2) -> bool:
        self._running = True
        self._loop.join(timeout_sec)
        is_stopped = not self._loop.is_alive()
        return is_stopped

    def _run(self) -> None:
        while self._running:
            pass

