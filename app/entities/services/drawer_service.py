from typing import Tuple
import numpy as np


class DrawerService:
    # --------------------------
    # Transform helpers
    # --------------------------
    def _rgb_to_hex(self, player_color: np.ndarray | list | None) -> str:
        if player_color is None:
            return "#A41D46"  # fallback color

        try:
            arr = np.array(player_color, dtype=float)
            arr = np.clip(arr, 0, 255).astype(int)
            return f"#{arr[0]:02x}{arr[1]:02x}{arr[2]:02x}"
        except Exception:
            return "#A41D46"

    def _scale_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Escala coordenadas del espacio 0-20/0-70 al sistema StatsBomb 120x80."""
        return x * 6, y * (80 / 70)
