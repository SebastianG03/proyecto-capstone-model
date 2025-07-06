from typing import Any, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from layers.infraestructure.video_analysis.plotting.diagram import Diagram
from layers.infraestructure.video_analysis.plotting.drawer_service import DrawerService

class VoronoiDiagramDrawer(Diagram):
    def __init__(self, players_tracks: List[Dict[int, Dict[str, Any]]]):  # Cambiado a lista de frames
        self.players_tracks = players_tracks
        self.home_team_color = 'blue'
        self.rival_team_color = 'red'
        self.save_path = '../app/res/output_videos/voronoi_diagram.png'
        self.drawer_service = DrawerService()

    def draw_and_save(self) -> None:
        self._draw_voronoi_diagram()

    def _draw_voronoi_diagram(self) -> None:
        # Procesar solo el primer frame para el ejemplo
        # (puedes modificarlo para procesar todos los frames)
        home_df, rival_df = pd.DataFrame(), pd.DataFrame()

        home, rival = self.drawer_service.process_frame(self.players_tracks[0])

        for frame in self.players_tracks:
            home_players, rival_players = self.drawer_service.process_frame(frame)
            home_df = pd.concat([home_df, home_players], join = 'inner', ignore_index=False)
            rival_df = pd.concat([rival_df, rival_players], join ='inner', ignore_index=False)

        
        if home_df.empty and rival_df.empty:
            return

        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor('white')
        ax.patch.set_facecolor('white')

        pitch = Pitch(
            pitch_type='statsbomb', 
            pitch_color='black',#'grass', 
            line_color='white',
            positional=True, 
            axis=True, 
            label=True, 
            tick=True, 
            #stripe=True
        )
        pitch.draw(ax=ax)

        # Combinar todos los jugadores para Voronoi
        all_players = pd.concat([home_df, rival_df])
        home_team, rival_team = pitch.voronoi(
            all_players.x, 
            all_players.y, 
            all_players.team
        )

        # Obtener colores únicos por equipo
        home_color =  home_df['color'] if not home_df.empty else self.home_team_color
        rival_color = rival_df['color'] if not rival_df.empty else self.rival_team_color

        print(f"Colors of voronoi: {home_color} -> home, and rival -> {rival_color}")

        pitch.polygon(home_team, ax=ax, fc=home_color, ec='white', lw=2, alpha=0.5)
        pitch.polygon(rival_team, ax=ax, fc=rival_color, ec='white', lw=2, alpha=0.5)
        
        plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        plt.close()