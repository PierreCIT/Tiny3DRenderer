import logging
from pathlib import Path
from rich.logging import RichHandler
from time import monotonic_ns

from renderer.renderer import Renderer, RenderMode
from renderer.color import Color, ColorMap
from renderer.tool import write_numpy_to_png


high_res = Renderer((4096, 2160), RenderMode.WIREFRAME)
log: logging.Logger | None = None

def setup_logging() -> None:
    format = "[%(asctime)s %(filename)-10s %(funcName)-15s:%(lineno)-4s] [%(levelname)-7s] %(message)s"
    path_to_log_file = Path(__file__).parent / "output/perf.log"
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path_to_log_file, mode='a', encoding=None, delay=False, errors=None)
    logging.basicConfig(format=format, level=logging.DEBUG, handlers=[RichHandler(), file_handler])
    global log
    log = logging.getLogger(__name__)
    log.info(f"Logger started. Log file can be found: {path_to_log_file}")

def perf_draw_lines_test() -> None:
    log.info("'perf_draw_lines_test': Starting...")
    im_size = high_res.image.shape
    time_spent = 0
    for l in range(im_size[0]):
        a = monotonic_ns()
        high_res.draw_line((l,0),  (l, im_size[1] - 1), Color())
        time_spent += monotonic_ns() - a
    log.info(f"'perf_draw_lines_test': Total time: {time_spent / 1e9}s,"
             f" {im_size[0]} lines, {im_size[1]} length ")

if __name__ == "__main__":
    setup_logging()
    perf_draw_lines_test()