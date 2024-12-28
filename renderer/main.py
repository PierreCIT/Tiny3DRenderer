import argparse
import logging
import numpy as np
import pyglet as pg
from pyglet.window import mouse, key
from time import monotonic_ns

from pathlib import Path
from rich.logging import RichHandler


from .config import Config
from .geometry import Pose
from .tool import write_numpy_to_png

log: logging.Logger | None = None

def setup_logging() -> None:
    format = "[%(asctime)s %(filename)-10s %(funcName)-15s:%(lineno)-4s] [%(levelname)-7s] %(message)s"
    path_to_log_file = Path(__file__).parent / "../logs/renderer.log"
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path_to_log_file, mode='a', encoding=None, delay=False, errors=None)
    logging.basicConfig(format=format, level=logging.DEBUG, handlers=[RichHandler(), file_handler])
    global log
    log = logging.getLogger(__name__)
    log.info(f"Logger started. Log file can be found at: {path_to_log_file}")

def main(config: Path, scene_name: str, interactive: bool= False) -> None:
    conf = Config(config)
    log.info("Scene parsed")
    scene = conf.load(scene_name)
    log.info("Scene loaded")
    if interactive:
        # Camera look at origin
        # for _, c in scene.cameras.items():
        #     c.lootak = Pose()

        img = scene.draw()
        w, h = img.shape[1], img.shape[0]
        window_name = f"Renderer {scene_name}: "
        window = pg.window.Window(w, h, caption=window_name)
        img = pg.image.ImageData(w, h, "RGBA", np.flipud(img.data).tobytes())
        # batch = pg.graphics.Batch()
        # frame = pg.gui.Frame(window, order=4)
        #
        # def set_cam_to_look_at() -> None:
        #     log.debug(f"Set cam look at")
        #     for _, c in scene.cameras.items():
        #         c.lootak = Pose()
        #
        # def unset_cam_to_look_at() -> None:
        #     log.debug(f"Unset cam look at")
        #     for _, c in scene.cameras.items():
        #         c.lootak = None
        #
        # check_img = pg.image.load((Path(__file__).parent / "../gui/imgs/check.png").resolve().as_posix())
        # checked_img = pg.image.load((Path(__file__).parent / "../gui/imgs/checked.png").resolve().as_posix())
        # toggle_button = pg.gui.ToggleButton(x=300, y=300, pressed=check_img, depressed=checked_img, batch=batch)
        # toggle_button.set_handler('on_toggle', set_cam_to_look_at)
        # frame.add_widget(pushbutton)
        # window.push_handlers(pushbutton)

        @window.event
        def on_draw():
            window.clear()
            fmt = "RGBA"
            pitch = w * len(fmt)
            before_render = monotonic_ns()
            render = scene.draw()
            render_time = (monotonic_ns() - before_render) / 1e9
            img.set_data("RGBA", pitch, data=np.flipud(render).tobytes())
            img.blit(0, 0)
            window.set_caption(f"{window_name}: {round(render_time, 2)}s (FPS: {round(1 / render_time, 2)})")
            #batch.draw()

        @window.event
        def on_mouse_press(x, y, button, modifiers):
            # Check which mouse button was pressed
            if button == mouse.LEFT:
                print("Left mouse button pressed")
            elif button == mouse.RIGHT:
                print("Right mouse button pressed")

        @window.event
        def on_key_press(symbol, modifiers):
            cam_move_speed = 0.5
            if symbol == key.UP:
                scene.cameras[list(scene.cameras.keys())[0]].move_z(cam_move_speed)
            elif symbol == key.DOWN:
                scene.cameras[list(scene.cameras.keys())[0]].move_z(-cam_move_speed)
            elif symbol == key.LEFT:
                scene.cameras[list(scene.cameras.keys())[0]].move_x(-cam_move_speed)
            elif symbol == key.RIGHT:
                scene.cameras[list(scene.cameras.keys())[0]].move_x()
            log.debug(f"New camera pose: {scene.cameras[list(scene.cameras.keys())[0]].pose}")

        pg.app.run()
    else:
        write_numpy_to_png(scene.draw(), Path(__file__).parent / "../images/render_0.png")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        prog='3D renderer',
        description='Render .obj files. Resources: https://github.com/rougier/tiny-renderer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s",
                        "--scene",
                        default="default",
                        type=str,
                        help="Scene name to load. Defined in configuration.")
    parser.add_argument("-c",
                        "--config",
                        default=Path(__file__).parent / "conf/default.json",
                        type=Path,
                        help="Path to the configuration file (.json).")
    parser.add_argument("-i",
                        "--interactive",
                        default=False,
                        action="store_true",
                        help="Specifiy if the rendering will be done in an interactive window.")

    args = parser.parse_args()
    main(args.config, args.scene, args.interactive)


