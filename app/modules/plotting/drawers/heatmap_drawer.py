from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from sqlalchemy.orm import Session

from app.modules.plotting.interfaces import Diagram
from app.entities.models import PlayerStateModel, HeatmapPointModel
from app.utils.routes import OUTPUT_VIDEOS_DIR


class HeatmapDrawer(Diagram):
    """
    Genera heatmaps por jugador y equipo (1/2) combinando:
      - PlayerStateModel (BD)
      - positions_history  {track_id: [np.array([x, y]), …]}
    Utiliza coordenadas transformadas cuando existan; sino (x, z).
    """

    def __init__(self, db: Session, positions: Dict[int, List[Optional[np.ndarray]]]):
        super().__init__(db)

        base = OUTPUT_VIDEOS_DIR
        self.save_path = base
        self.players_path = base / "players"
        self.positions_history = positions  # {track_id: [np.ndarray([x, y]), …]}

        for p in [base, self.players_path]:
            p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _safe_concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame(columns=["x", "y"])
        df = pd.concat(dfs, ignore_index=True)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)
        return df

    def _is_valid_for_kde(self, df: pd.DataFrame) -> bool:
        if df.empty or len(df) < 5:
            return False
        if df["x"].min() == df["x"].max() or df["y"].min() == df["y"].max():
            return False
        return True

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

    # ------------------------------------------------------------------
    # BD + positions_history
    # ------------------------------------------------------------------
    def _build_player_df(self, player_id: int) -> pd.DataFrame:
        """
        1. Lee estados del jugador desde BD
        2. Agrega / sobre-escribe con positions_history
        3. Devuelve DataFrame con (x, y, team) donde team ∈ {"1","2"}
        """
        states: List[PlayerStateModel] = (
            self.db.query(PlayerStateModel)
            .filter(PlayerStateModel.player_id == player_id)
            .order_by(PlayerStateModel.frame_index)
            .all()
        )

        rows: List[dict] = []
        used_frames: Set[int] = set()

        # Primero metemos lo que tenemos en BD
        for st in states:
            # elegimos coordenadas
            if st.x_transformed is not None and st.y_transformed is not None:
                xx, yy = float(f'{st.x_transformed}'), float(f'{st.y_transformed}')
            else:
                xx, yy = float(f'{st.x}'), float(f'{st.y}')

            if xx is None or yy is None:
                continue

            # team siempre 1 o 2
            team_str = str(st.team) if st.team in {"1", "2"} else "1"

            rows.append(
                {
                    "frame": st.frame_index,
                    "x": float(xx),
                    "y": float(yy),
                    "team": team_str,
                }
            )
            used_frames.add(int(f'{st.frame_index}'))

        # 2) Complementamos con positions_history
        if player_id in self.positions_history:
            # Buscamos equipo a partir del último state conocido
            default_team = str(states[-1].team) if states else "1"
            default_team = default_team if default_team in {"1", "2"} else "1"

            for idx, pos in enumerate(self.positions_history[player_id]):
                if pos is None or len(pos) < 2:
                    continue
                # Si ya existe en BD para este frame, skip (prioridad BD)
                if idx in used_frames:
                    continue
                rows.append(
                    {
                        "frame": idx,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "team": default_team,
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def draw_and_save(self) -> None:
        print("Generando heatmaps combinando BD + positions_history…")
        players = (
            self.db.query(PlayerStateModel.player_id, PlayerStateModel.team)
            .distinct()
            .filter(PlayerStateModel.player_id.is_not(None))
            .all()
        )
        teams = {f"{player.team}" for player in players}
        player_ids: List[int] = [int(p.player_id) for p in players]
        print(f"{len(player_ids)} jugadores encontrados.")

        for pid in player_ids:
            print(f"Procesando jugador {pid}…")
            df = self._build_player_df(pid)
            if df.empty:
                print(f"Sin datos para {pid}, skip.")
                continue

            # Dibujamos un png por equipo
            for team_str in teams:
                team_df = df[df["team"] == team_str][["x", "y"]]
                if not self._is_valid_for_kde(team_df):
                    print(f"  equipo {team_str} sin datos suficientes")
                    continue
                self._draw_single_heatmap(pid, team_str, team_df)

    # ------------------------------------------------------------------
    # draw 1 archivo
    # ------------------------------------------------------------------
    def _draw_single_heatmap(self, pid: int, team: str, df: pd.DataFrame):
        fig, ax, pitch = self._draw_pitch()

        levels = min(60, max(10, len(df) // 2))
        pitch.kdeplot(
            df["x"],
            df["y"],
            ax=ax,
            cmap="viridis",
            fill=True,
            alpha=0.6,
            levels=levels,
            bw_adjust=0.3,
        )

        out_file = self.players_path / f"heatmap_player_{pid}_team_{team}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  --> guardado {out_file}")
