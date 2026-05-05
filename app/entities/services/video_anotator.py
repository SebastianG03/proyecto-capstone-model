from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.utils.routes import ANOTATED_OUTPUT_IMAGES
from app.core.config import DEBUG

class VideoAnotator:
    """
    Maneja:
    - Anotaciones visuales
    - Escritura del video
    - Preview en tiempo real
    - Liberacion segura de recursos
    """

    def __init__(
        self,
        output_path: Path,
        fps,
        colors: dict,
        show_preview: bool = True,
        window_name: str = "Annotated Video"
    ):
        try:
            self.colors = colors
            self.show_preview = show_preview
            self.window_name = window_name
            self.is_closed = False
            self.frame_size = (640,640)

            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                output_path.as_posix(),
                fourcc,
                fps,
                self.frame_size
            )

            if self.show_preview:
                cv2.namedWindow(self.window_name, cv2.WINDOW_FULLSCREEN)
        except RuntimeError as runtime_error:
            raise runtime_error
        except Exception as error:
            raise error


    def annotate(self, frame: np.ndarray, bbox: list[float], class_name: str, text: str):

        annotated = frame.copy()
        # H, W = frame.shape[:2]

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = self.colors.get(class_name, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # label background
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - th - 8),
            (x1 + tw, y1),
            color,
            -1
        )

        cv2.putText(
            annotated,
            text,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

        return annotated

    def save_frame(self, frame, frame_num: int):
        """
        Saves the annotated frame to a file named "annotated_frame_<uuid>.jpg"

        Parameters
        ----------
        frame : np.ndarray
            The annotated frame to be saved
        """
        output_path = ANOTATED_OUTPUT_IMAGES / f"annotated_frame_{frame_num}_{uuid4()}.jpg"
        cv2.imwrite(output_path.as_posix(), frame)

    def write(self, frame: np.ndarray, frame_num: int):
        if DEBUG:
            self.save_frame(frame, frame_num)
        if self.is_closed:
            return

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]

        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return

        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)

    def show(self, frame):
        """
        Muestra frame en pantalla.
        Retorna False si usuario pide salir.
        """
        if not self.show_preview:
            return True

        cv2.imshow(self.window_name, frame)

        # waitKey es obligatorio para refrescar ventana
        key = cv2.waitKey(1) & 0xFF

        # ESC o Q para salir
        if key in (27, ord("q")):
            return False

        return True

    def write_and_show(self, frame, frame_num: int):
        """
        Metodo recomendado.
        """
        self.write(frame, frame_num)
        return self.show(frame)

    def close(self):

        if self.is_closed:
            return

        self.writer.release()

        if self.show_preview:
            cv2.destroyWindow(self.window_name)

        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

