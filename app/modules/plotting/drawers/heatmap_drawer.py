from pathlib import Path
from typing import Dict, List, Optional, Set, override
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from sqlalchemy.orm import Session

from app.entities.models.PlayerModels import Player
from app.modules.plotting.interfaces import Diagram
from app.entities.models import PlayerState
from app.utils.routes import OUTPUT_VIDEOS_DIR


class HeatmapDrawer(Diagram):
    """
    Genera heatmaps por jugador/equipo **únicamente** con datos de BD
    (PlayerState + Player).  Rellena saltos extremos para evitar
    discontinuidades en el KDE.
    """

    def __init__(self, db: Session):
        """
        Inicializa la clase HeatmapDrawer.

        Args:
            db (Session): Sesión de base de datos.
            positions (Dict[int, List[Optional[np.ndarray]]]):
                Diccionario que mapea player_id a lista de posiciones
                (x, y) en el frame correspondiente.
        """
        super().__init__(db)

        base = OUTPUT_VIDEOS_DIR
        self.save_path = base
        self.players_path = base / "players"

        for p in [base, self.players_path]:
            p.mkdir(parents=True, exist_ok=True)

    def _safe_concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatena una lista de DataFrames en un solo DataFrame, evitando
        discontinuidades en el KDE.

        Args:
            dfs (List[pd.DataFrame]): Lista de DataFrames a concatenener.

        Returns:
            pd.DataFrame: DataFrame concatenado y limpio de valores no
            numéricos y nulos.
        """
        if not dfs:
            return pd.DataFrame(columns=["x", "y"])
        df = pd.concat(dfs, ignore_index=True)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)
        return df

    def _is_valid_for_kde(self, df: pd.DataFrame) -> bool:
        """
        Verifica si un DataFrame es válido para ser utilizado en un
        kernel density estimation (KDE).

        Un DataFrame es válido si no está vacío y tiene al menos 3
        puntos.

        Args:
            df (pd.DataFrame): DataFrame a verificar.

        Returns:
            bool: True si el DataFrame es válido, False en caso contrario.
        """
        return not df.empty and len(df) >= 3  # mínimo 3 puntos

    def _draw_pitch(self):
        """
        Dibuja un pitch de fútbol con la configuración por defecto.

        Returns:
            fig (matplotlib.figure.Figure): Figura del pitch.
            ax (matplotlib.axes.Axes): Eje del pitch.
            pitch (Pitch): Instancia de Pitch con la configuración por defecto.
        """

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
        Construye un DataFrame con las coordenadas (x, y) de un jugador,
        rellenando saltos extremos con puntos intermedios.

        Args:
            player_id (int): Identificador del usuario.

        Returns:
            pd.DataFrame: DataFrame con las coordenadas (x, y) del usuario
            y su equipo, rellenando saltos extremos con puntos intermedios.
        """
        states: List[PlayerState] = (
            self.db.query(PlayerState)
            .filter(PlayerState.player_id == player_id)
            .order_by(PlayerState.frame_index)
            .all()
        )
        if not states:
            return pd.DataFrame(columns=["x", "y", "team"])

        # obtener equipo del jugador
        player = (
            self.db.query(Player)
            .filter(Player.player_id == player_id)
            .first()
        )
        team = player.team if player and f'{player.team}' else "None"

        rows: List[dict] = []
        for st in states:
            # coordenadas preferidas: transformed > raw
            xx = float(f'{st.x_transformed}') if st.x_transformed is not None else float(f'{st.x}')
            yy = float(f'{st.y_transformed}') if st.y_transformed is not None else float(f'{st.y}')
            if xx is None or yy is None:
                continue
            rows.append({"x": float(xx), "y": float(yy), "team": team})

        df = pd.DataFrame(rows)
        if len(df) < 2:
            return df

        # --------- rellenar saltos extremos ------------------------------
        # Umbral: mitad del ancho del campo (StatsBomb: 120 unidades)
        MAX_JUMP = 60.0
        fill_dfs: List[pd.DataFrame] = [df.iloc[[0]]]

        for i in range(1, len(df)):
            prev = fill_dfs[-1].iloc[-1]
            curr = df.iloc[i]

            dist = np.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
            if dist > MAX_JUMP:
                # generar puntos intermedios (peso bajo)
                n_pts = max(2, int(dist / MAX_JUMP))
                x_vals = np.linspace(prev["x"], curr["x"], n_pts + 1)[1:-1]
                y_vals = np.linspace(prev["y"], curr["y"], n_pts + 1)[1:-1]
                fill_df = pd.DataFrame(
                    {"x": x_vals, "y": y_vals, "team": team}
                )
                fill_dfs.append(fill_df)

            fill_dfs.append(df.iloc[[i]])

        df_filled = pd.concat(fill_dfs, ignore_index=True)
        return df_filled

    @override
    def draw_and_save(self) -> None:
        """
        Genera un heatmap por jugador y equipo.
        Primero, extrae los datos de posición de cada jugador y equipo.
        Luego, dibuja un heatmap por cada equipo (solo debería haber 1).
        """

        print("Generando heatmaps (solo BD)…")
        player_ids = [
            int(p[0])
            for p in self.db.query(PlayerState.player_id).distinct().all()
            if p[0] is not None
        ]
        print(f"{len(player_ids)} jugadores encontrados.")

        for pid in player_ids:
            print(f"Procesando jugador {pid}…")
            df = self._build_player_df(pid)
            if not self._is_valid_for_kde(df):
                print(f"  ↳ datos insuficientes (<3) – skip")
                continue

            # un png por equipo (solo debería haber 1)
            for team in df["team"].unique():
                team_df = df[df["team"] == team][["x", "y"]]
                self._draw_single_heatmap(pid, str(team), team_df)


    def _draw_single_heatmap(self, pid: int, team: str, df: pd.DataFrame):
        """
        Dibuja un heatmap para un solo jugador y equipo.
        """
        # ---------- 1) depuración ----------
        print(f"  jugador {pid}  n={len(df)}  "
            f"x∈[{df.x.min():.1f}, {df.x.max():.1f}]  "
            f"y∈[{df.y.min():.1f}, {df.y.max():.1f}]")

        # ---------- 2) recortar al campo ----------
        PITCH_X, PITCH_Y = [0, 120], [0, 80]
        df = df[df.x.between(PITCH_X[0], PITCH_X[1]) & df.y.between(PITCH_Y[0], PITCH_Y[1])]
        if df.empty:
            print("  ↳ ninguna coordenada dentro del campo – skip")
            return

        # ---------- 3) evitar desvío nulo ----------
        if df.x.nunique() == 1 or df.y.nunique() == 1:
            print("  ↳ todos los puntos idénticos – skip")
            return

        fig, ax, pitch = self._draw_pitch()

        # ---------- 4) bw dinámico ----------
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
