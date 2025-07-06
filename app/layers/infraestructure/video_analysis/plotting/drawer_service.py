from typing import Any, Dict, List

import pandas as pd


class DrawerService():
    def _rgb_to_hex(self, player_color: List[float]) -> str:
        r = int(round(player_color[0]))
        g = int(round(player_color[1]))
        b = int(round(player_color[2]))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _scale_coordinates(self, x: float, y: float) -> tuple:
        """Escala las coordenadas al sistema de StatsBomb (120x80)"""
        # Escala X: de 0-20 a 0-120
        # Escala Y: de 0-70 a 0-80
        scaled_x = x * 6  # 20 * 6 = 120
        scaled_y = y * (80/70)  # 70 * (80/70) = 80
        return scaled_x, scaled_y

    def process_frame(self, frame: Dict[int, Dict[str, Any]]):
        """Procesa un frame y devuelve DataFrames escalados"""
        home_players = []
        rival_players = []
        
        for player_id, track in frame.items():
            if track['position_transformed'] is None:
                continue
                
            # Escalar coordenadas
            x, y = self._scale_coordinates(*track['position_transformed'])
            
            player_data = {
                'x': x,
                'y': y,
                'id': player_id,
                'team': track['team'],
                'color': self._rgb_to_hex(track['team_color'])
            }
            
            
            if track['team'] == 1:
                home_players.append(player_data)
            else:
                rival_players.append(player_data)
        
        
        return pd.DataFrame(home_players), pd.DataFrame(rival_players)