from typing import List, override
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from sqlalchemy.orm import Session

from app.entities.models.PlayerModels import Player
from app.entities.interfaces import Diagram
from app.entities.models import PlayerState
import app.entities.utils.tools_context as context 
from app.utils.routes import OUTPUT_VIDEOS_DIR
from app.logger import debug_logger

RAW_W, RAW_H = 1280, 720  # pixeles de la fuente
PITCH_X, PITCH_Y = [0, 120], [0, 80]  # unidades StatsBomb


def _normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza x e y de pixeles a 0-120 y 0-80 respectivamente.
    Se hace una copia para no modificar el original.
    """
    df = df.copy()
    df["x"] = (df["x"] / RAW_W) * (PITCH_X[1] - PITCH_X[0]) + PITCH_X[0]
    df["y"] = (df["y"] / RAW_H) * (PITCH_Y[1] - PITCH_Y[0]) + PITCH_Y[0]
    return df


class HeatmapDrawer(Diagram):
    """
    Genera heatmaps por jugador/equipo **unicamente** con datos de BD
    (PlayerState + Player).  Rellena saltos extremos para evitar
    discontinuidades en el KDE.
    """

    def __init__(self, db: Session):
        super().__init__(db)
        self.tools = context.analysis_context.tools
        base = OUTPUT_VIDEOS_DIR
        self.players_path = base / "players"
        for p in [base, self.players_path]:
            p.mkdir(parents=True, exist_ok=True)

    # --- resto de metodos auxiliares sin cambios ---
    def _safe_concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame(columns=["x", "y"])
        df = pd.concat(dfs, ignore_index=True)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)
        return df

    def _draw_pitch(self):
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1e4251",
            line_color="white",
            axis=True,
            label=True,
        )
        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        pitch.draw(ax=ax)
        return fig, ax, pitch

    def _build_player_df(self, player_id: int) -> pd.DataFrame:
        """
        Construye DataFrame con coordenadas NORMALIZADAS usando timestamp_ms
        como orden cronologico (no frame_index).
        """
        # 1. leemos por timestamp_ms  ‹‹‹ CAMBIO
        states: List[PlayerState] = (
            self.db
            .query(PlayerState)
            .filter(PlayerState.player_id == player_id)
            .order_by(PlayerState.timestamp_ms)  # ← antes frame_index
            .all()
        )
        if not states:
            return pd.DataFrame(columns=["x", "y", "team"])

        player = self.db.query(Player).filter(Player.player_id == player_id).first()
        team = player.team if player and f"{player.team}" else "None"

        rows = []
        for st in states:
            xx = (
                float(f"{st.x_transformed}")
                if st.x_transformed is not None
                else float(f"{st.x}")
            )
            yy = (
                float(f"{st.y_transformed}")
                if st.y_transformed is not None
                else float(f"{st.y}")
            )
            if pd.isna(xx) or pd.isna(yy):
                continue
            rows.append({"x": float(xx), "y": float(yy), "team": team})

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # NORMALIZAR
        df = _normalize_coords(df)
        if len(df) < 2:
            return df

        # rellenar saltos
        MAX_JUMP = 60.0
        fill_dfs = [df.iloc[[0]]]
        for i in range(1, len(df)):
            prev = fill_dfs[-1].iloc[-1]
            curr = df.iloc[i]
            dist = np.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
            if dist > MAX_JUMP:
                n_pts = max(2, int(dist / MAX_JUMP))
                x_vals = np.linspace(prev["x"], curr["x"], n_pts + 1)[1:-1]
                y_vals = np.linspace(prev["y"], curr["y"], n_pts + 1)[1:-1]
                fill_dfs.append(pd.DataFrame({"x": x_vals, "y": y_vals, "team": team}))
            fill_dfs.append(df.iloc[[i]])

        return pd.concat(fill_dfs, ignore_index=True)

    def _is_valid_for_kde(self, df: pd.DataFrame) -> bool:
        # Permite dibujar un punto unico o kde con 3+
        return not df.empty and len(df) >= 1

    @override
    def draw_and_save(self) -> None:
        print("Generando heatmaps (solo BD)…")
        # Todos los player_id que existan en la tabla Player
        player_ids = [
            int(p[0])
            for p in self.db.query(Player.player_id).distinct().all()
            if p[0] is not None
        ]
        print(f"{len(player_ids)} jugadores encontrados en tabla Player.")

        for pid in player_ids:
            print(f"Procesando jugador {pid}…")
            df = self._build_player_df(pid)

            if df.empty:
                # Usamos team="None" para el nombre del archivo
                self._draw_single_heatmap(pid, "None", pd.DataFrame(columns=["x", "y"]))
                continue

            for team in df["team"].unique():
                team_df = df[df["team"] == team][["x", "y"]]
                self._draw_single_heatmap(pid, str(team), team_df)

    def _draw_single_heatmap(self, pid: int, team: str, df: pd.DataFrame):
        print(
            f"  jugador {pid}  n={len(df)}  "
            f"x∈[{df.x.min():.1f}, {df.x.max():.1f}]  "
            f"y∈[{df.y.min():.1f}, {df.y.max():.1f}]"
            if not df.empty
            else f"  jugador {pid}  SIN DATOS – heatmap vacio"
        )

        fig, ax, pitch = self._draw_pitch()

        if df.empty:
            # Nada mas que el campo limpio
            print("  ↳ DataFrame vacio – campo limpio")
        elif df.x.nunique() == 1 and df.y.nunique() == 1:
            # Punto unico
            ax.scatter(
                df.x.iloc[0],
                df.y.iloc[0],
                s=600,
                c="red",
                edgecolors="white",
                linewidths=2,
                zorder=5,
            )
            print("  ↳ punto unico dibujado")
        else:
            # KDE normal
            bw = max(0.5, 4 / np.sqrt(len(df)))
            pitch.kdeplot(
                df.x,
                df.y,
                ax=ax,
                cmap="viridis",
                fill=True,
                alpha=0.6,
                levels=30,
                bw_adjust=bw,
            )

        out_file = self.players_path / f"heatmap_player_{pid}_team_{team}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  --> guardado {out_file}")
        payload = {"player_id": pid, "path": out_file.as_posix()}
        debug_logger.debug(f"[HeatmapDrawer] Enviando payload: {payload}")
        self.tools.heatmap_points.post(payload)
        return out_file

