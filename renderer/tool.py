import numpy as np
from logging import getLogger
from pathlib import Path
from PIL import Image

log = getLogger(__name__)

def write_numpy_to_png(arr:np.ndarray, f: Path) -> None:
    f = f.resolve().absolute().as_posix()
    if f.endswith(".png"):
        Image.fromarray(arr, mode="RGBA").save(f)
    else:
        log.warning(f"File format may not be handled for: '{f}'")
        Image.fromarray(arr, mode="RGBA").save(f)