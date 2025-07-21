# layers/infraestructure/video_analysis/plotting/improved_ball_trajectory_drawer.py

from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np
import matplotlib.gridspec as gridspec

class BallTrajectoryDrawer:
    def __init__(self, players_tracks: List):
        self.players_tracks = players_tracks
        self.base_save_path = '../app/res/output_videos/'
        self.ball_positions: List = []
        self.team_colors_map: Dict = {}
        self.player_names = {}
        self.df = None
        self.segments = []

    def draw_and_save(self) -> None:
        self._collect_ball_positions()
        if not self.ball_positions:
            print("No se encontraron datos de posesión del balón.")
            return
        self._generate_player_names()
        self._prepare_data()
        self._draw_summary_diagram()
        self._draw_table_diagram()
        self._draw_minimaps_diagram()

    def _collect_ball_positions(self) -> None:
        for frame_index, frame in enumerate(self.players_tracks):
            ball_position = self._get_ball_position(frame, frame_index)
            if ball_position:
                x, y, player_id, team_id = ball_position
                self.ball_positions.append((x, y, player_id, team_id, frame_index))
                if team_id not in self.team_colors_map:
                    for player_data in frame.values():
                        if player_data.get('team') == team_id:
                            color_arr = player_data['team_color']
                            self.team_colors_map[team_id] = tuple(c / 255 for c in color_arr)
                            break

    def _get_ball_position(self, frame: Dict[int, Dict[str, Any]], frame_index: int
                          ) -> Optional[Tuple[float, float, int, int]]:
        for player_id, player_data in frame.items():
            if player_data.get('has_ball', False):
                pos = player_data.get('position_transformed')
                if pos is None:
                    continue
                return (pos[0], pos[1], player_id, player_data['team'])
        return None

    def _generate_player_names(self) -> None:
        player_ids = set(point[2] for point in self.ball_positions)
        self.player_names = {pid: f"Jugador {pid}" for pid in player_ids}
        key_players = {1: "Delantero", 10: "Mediocampista", 20: "Portero"}
        for pid, name in key_players.items():
            if pid in self.player_names:
                self.player_names[pid] = name

    def _prepare_data(self) -> None:
        self.df = pd.DataFrame(self.ball_positions, 
                              columns=['x', 'y', 'player_id', 'team', 'frame_index'])
        self.segments = self._split_trajectory_into_segments(self.df)

    def _split_trajectory_into_segments(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        segments = []
        start_idx = 0
        for i in range(1, len(df)):
            if df.iloc[i].player_id != df.iloc[i-1].player_id or i % 50 == 0:
                segments.append(df.iloc[start_idx:i])
                start_idx = i
        segments.append(df.iloc[start_idx:])
        return segments

    def _draw_summary_diagram(self) -> None:
        fig = plt.figure(figsize=(12, 8))
        fig.set_facecolor('#1e4251')
        ax = fig.add_subplot(111)
        
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
        
        # Dibujar trayectoria base
        if len(self.df) > 1:
            ax.plot(self.df.x, self.df.y, 'w-', alpha=0.3, lw=1)
        
        # Dibujar segmentos desplazados
        for i, segment in enumerate(self.segments):
            if len(segment) < 2:
                continue
                
            center_x, center_y = segment.iloc[0].x, segment.iloc[0].y
            displacement = i * 1.5
            x_displaced = []
            y_displaced = []
            
            for _, row in segment.iterrows():
                dx = row.x - center_x
                dy = row.y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                new_dist = dist + displacement
                x_displaced.append(center_x + new_dist * np.cos(angle))
                y_displaced.append(center_y + new_dist * np.sin(angle))
            
            color = self.team_colors_map[segment.iloc[0].team]
            ax.plot(x_displaced, y_displaced, '-', color=color, lw=2, alpha=0.8)
            
            # Marcadores de inicio/fin
            ax.scatter(x_displaced[0], y_displaced[0], s=100, 
                      color=color, edgecolor='white', zorder=10)
            ax.scatter(x_displaced[-1], y_displaced[-1], s=100, 
                      color=color, marker='s', edgecolor='white', zorder=10)
            
            player_id = segment.iloc[0].player_id
            label = f"{self.player_names[player_id]} ({player_id})"
            ax.text(x_displaced[0], y_displaced[0] + 2, label,
                   color='white', fontsize=8, ha='center',
                   bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.2'))
        
        plt.title('Trayectoria Principal del Balón', color='white', fontsize=16)
        plt.savefig(f'{self.base_save_path}ball_trajectory_summary.png', 
                   dpi=300, bbox_inches='tight', facecolor='#1e4251')
        plt.close()

    def _draw_table_diagram(self) -> None:
        # Calcular posesión - USAR ACCESO POR LLAVES
        possession = self.df.groupby('player_id').size().reset_index(name='count')
        possession['percentage'] = (possession['count'] / len(self.df)) * 100
        possession = possession.sort_values('percentage', ascending=False)
        
        # Preparar datos de la tabla CORRECTAMENTE
        table_data = []
        team_colors = []
        
        for _, row in possession.iterrows():
            player_id = row['player_id']  # Acceso correcto
            count = row['count']           # Acceso correcto
            percentage = row['percentage'] # Acceso correcto
            
            player_name = self.player_names[player_id]
            team_id = self.df[self.df['player_id'] == player_id].iloc[0]['team']
            team_color = self.team_colors_map[team_id]
            
            team_colors.append(team_color)
            table_data.append([
                f"{player_name} ({player_id})",
                f"{count} frames",  # Valor numérico real
                f"{percentage:.1f}%"  # Valor numérico real
            ])
        
        # Crear figura
        fig = plt.figure(figsize=(10, 4))
        fig.set_facecolor('#1e4251')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Crear tabla
        table = ax.table(
            cellText=table_data,
            colLabels=['Jugador', 'Posesión', 'Porcentaje'],
            loc='center',
            cellLoc='center'
        )
        
        # Formatear tabla
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        for i, key in enumerate(table.get_celld().keys()):
            cell = table.get_celld()[key]
            if key[0] == 0:
                cell.set_facecolor('#3a6b7e')
                cell.set_text_props(color='white', weight='bold')
            elif key[0] > 0:
                # Usar colores del equipo con transparencia
                cell.set_facecolor(team_colors[key[0]-1] + (0.7,))
                cell.set_text_props(color='white')
        
        plt.title('Distribución de Posesión del Balón', color='white', fontsize=14, pad=20)
        plt.savefig(f'{self.base_save_path}ball_trajectory_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='#1e4251')
        plt.close()

    def _draw_minimaps_diagram(self) -> None:
        n_segments = len(self.segments)
        n_cols = min(4, n_segments)
        n_rows = (n_segments + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 4 * n_rows))
        fig.set_facecolor('#1e4251')
        fig.suptitle('Segmentos de la Trayectoria', color='white', fontsize=16, y=0.95)
        
        for i, segment in enumerate(self.segments):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            self._draw_minimap(ax, segment, i)
        
        plt.savefig(f'{self.base_save_path}ball_trajectory_minimaps.png', 
                   dpi=300, bbox_inches='tight', facecolor='#1e4251')
        plt.close()

    def _draw_minimap(self, ax: plt.Axes, segment: pd.DataFrame, idx: int) -> None:
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color='#2a5b6e',
            line_color='white',
            linewidth=0.8,
            axis=False,
            label=False
        )
        pitch.draw(ax=ax)
        
        color = self.team_colors_map[segment.iloc[0].team]
        ax.plot(segment.x, segment.y, 'o-', color=color, markersize=3, lw=1, alpha=0.8)
        ax.scatter(segment.iloc[0].x, segment.iloc[0].y, s=30, color=color, edgecolor='white')
        ax.scatter(segment.iloc[-1].x, segment.iloc[-1].y, s=30, color=color, marker='s', edgecolor='white')
        
        start_frame = segment.iloc[0].frame_index
        end_frame = segment.iloc[-1].frame_index
        player_name = self.player_names[segment.iloc[0].player_id]
        ax.set_title(f"Seg {idx+1}: F {start_frame}-{end_frame}\n{player_name}", 
                    color='white', fontsize=10)