from typing import Dict, Tuple
import pandas as pd
import numpy as np
from app.layers.domain.tracks.track_detail import TrackDetailBase


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
            return f'#{arr[0]:02x}{arr[1]:02x}{arr[2]:02x}'
        except Exception:
            return "#A41D46"

    def _scale_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Escala coordenadas del espacio 0-20/0-70 al sistema StatsBomb 120x80."""
        return x * 6, y * (80 / 70)

    # --------------------------
    # FRAME PROCESSING
    # --------------------------
    def process_frame(self, frame: Dict[int, TrackDetailBase]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        home_players = []
        rival_players = []

        for player_id, track in frame.items():
            # Validación mínima
            if (
                track is None
                or track.position_transformed is None
                or len(track.position_transformed) < 2
            ):
                continue

            try:
                x_raw, y_raw = float(track.position_transformed[0]), float(track.position_transformed[1])
            except (ValueError, TypeError):
                continue

            x, y = self._scale_coordinates(x_raw, y_raw)

            player_data = {
                "id": player_id,
                "x": x,
                "y": y,
                "team": getattr(track, "team", -1),
                "color": self._rgb_to_hex(getattr(track, "team_color", None)),
            }

            if player_data["team"] == 1:
                home_players.append(player_data)
            else:
                rival_players.append(player_data)

        return pd.DataFrame(home_players), pd.DataFrame(rival_players)
