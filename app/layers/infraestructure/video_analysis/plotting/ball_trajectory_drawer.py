
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np

from layers.infraestructure.video_analysis.plotting.diagram import Diagram

class BallTrajectoryDrawer(Diagram):
    def __init__(self, players_tracks: List):
        self.players_tracks = players_tracks
        self.save_path = '../app/res/output_videos/ball_trajectory.png'
        self.ball_positions: List = []  # (x, y, player_id, team, frame_index)
        self.team_colors_map: Dict = {}

    def draw_and_save(self) -> None:
        self._collect_ball_positions()
        if not self.ball_positions:
            print("No se encontraron datos de posesión del balón.")
            return
        self._draw_trajectory()

    def _collect_ball_positions(self) -> None:
        """Recopila las posiciones del balón y asigna colores por equipo."""
        for frame_index, frame in enumerate(self.players_tracks):
            ball_position = self._get_ball_position(frame, frame_index)
            if ball_position:
                x, y, player_id, team_id = ball_position
                self.ball_positions.append((x, y, player_id, team_id, frame_index))
                
                # Asignar color al equipo si no está registrado
                if team_id not in self.team_colors_map:
                    player_data = next(p for p in frame.values() if p.get('team') == team_id)
                    color_arr = player_data['team_color']
                    self.team_colors_map[team_id] = tuple(c / 255 for c in color_arr)

    def _get_ball_position(self, frame: Dict[int, Dict[str, Any]], frame_index: int
                          ) -> Optional[Tuple[float, float, int, int]]:
        """Obtiene la posición del balón en un frame específico."""
        for player_id, player_data in frame.items():
            if player_data.get('has_ball', False):
                pos = player_data.get('position_transformed')
                if pos is None:
                    continue  # Saltar si no hay posición transformada
                return (pos[0], pos[1], player_id, player_data['team'])
        return None

    def _draw_trajectory(self) -> None:
        """Dibuja la trayectoria del balón y las posesiones de los jugadores."""
        df = pd.DataFrame(self.ball_positions, 
                         columns=['x', 'y', 'player_id', 'team', 'frame_index'])
        
        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor('white')
        ax.patch.set_facecolor('white')
        
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color='#1e4251',
            line_color='white',
            linewidth=1.5,
            axis=True,
            label=True,
            tick=True
        )
        pitch.draw(ax=ax)
        
        # Dibujar trayectoria principal
        if len(df) > 1:
            pitch.lines(
                df.x.iloc[:-1], df.y.iloc[:-1],
                df.x.iloc[1:], df.y.iloc[1:],
                ax=ax, color='white', lw=2, comet=True, alpha=0.6
            )
        
        # Colores para los puntos
        colors = [self.team_colors_map[team] for team in df.team]
        
        # Dibujar puntos de posición
        pitch.scatter(
            df.x, df.y, ax=ax, 
            c=colors, s=200, 
            edgecolors='black', zorder=3, alpha=0.8
        )
        
        # Etiquetar puntos clave
        self._label_key_points(ax, df)
        
        # Leyenda de equipos
        self._add_legend(ax, df)
        
        plt.title('Recorrido del Balón', color='white', fontsize=14)
        plt.savefig(self.save_path, dpi=300, bbox_inches='tight', facecolor='#1e4251')
        plt.close()

    def _label_key_points(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Etiqueta puntos clave en la trayectoria."""
        # Primer punto
        start = df.iloc[0]
        ax.text(
            start.x, start.y + 2, f"Inicio: J{start.player_id}",
            color='white', fontsize=9, ha='center', va='bottom',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
        )
        
        # Último punto
        end = df.iloc[-1]
        ax.text(
            end.x, end.y + 2, f"Fin: J{end.player_id}",
            color='white', fontsize=9, ha='center', va='bottom',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
        )
        
        # Cambios de posesión
        change_indices = []
        for i in range(1, len(df)):
            if df.iloc[i].player_id != df.iloc[i-1].player_id:
                change_indices.append(i)
        
        for idx in change_indices:
            point = df.iloc[idx]
            ax.text(
                point.x, point.y - 3, f"Cambio: J{point.player_id}",
                color='white', fontsize=8, ha='center', va='top',
                bbox=dict(facecolor='#e63946', alpha=0.9, boxstyle='round,pad=0.3')
            )

    def _add_legend(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Añade leyenda de equipos al gráfico."""
        unique_teams = df.team.unique()
        legend_labels = []
        
        for team_id in unique_teams:
            color = self.team_colors_map[team_id]
            # Crear marcador para la leyenda
            legend_labels.append(
                plt.Line2D([0], [0], 
                          marker='o', 
                          color='w', 
                          markerfacecolor=color,
                          markersize=10,
                          label=f'Equipo {team_id}')
            )
        
        ax.legend(
            handles=legend_labels, 
            loc='upper right',
            facecolor='#2a5b6e',
            edgecolor='none',
            labelcolor='white',
            title='Equipos',
            title_fontproperties={'weight': 'bold'}
        )