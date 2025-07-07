# layers/infraestructure/video_analysis/plotting/heatmap_drawer.py

from typing import Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from layers.infraestructure.video_analysis.plotting.diagram import Diagram
from layers.infraestructure.video_analysis.plotting.drawer_service import DrawerService


class HeatmapDrawer(Diagram):
    def __init__(self, players_tracks: List[Dict[int, Dict[str, Any]]]):
        self.players_tracks = players_tracks
        self.save_path = '../app/res/output_videos/heatmap.png'
        self.drawer_service = DrawerService()

    def draw_and_save(self) -> None:
        self._draw_heatmap()

    def _draw_heatmap(self) -> None:
        home_df, rival_df = pd.DataFrame(), pd.DataFrame()

        for frame in self.players_tracks:
            home_players, rival_players = self.drawer_service.process_frame(frame)
            home_df = pd.concat([home_df, home_players], ignore_index=True)
            rival_df = pd.concat([rival_df, rival_players], ignore_index=True)

        if home_df.empty and rival_df.empty:
            return

        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor('white')
        ax.patch.set_facecolor('white')

        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color='black',
            line_color='white',
            axis=True,
            label=True,
            tick=True
        )
        pitch.draw(ax=ax)

        # Dibujar heatmap de cada equipo
        if not home_df.empty:
            pitch.kdeplot(
                home_df.x, home_df.y,
                ax=ax, cmap='Blues', fill=True, alpha=0.6, levels=100, bw_adjust=0.3
            )
        if not rival_df.empty:
            pitch.kdeplot(
                rival_df.x, rival_df.y,
                ax=ax, cmap='Reds', fill=True, alpha=0.6, levels=100, bw_adjust=0.3
            )

        plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        plt.close()
