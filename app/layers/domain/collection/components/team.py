from typing import Any, Dict, List

import numpy as np
from app.layers.domain.player.player import Player


class Team():
    team: int
    team_name: str
    players: list[Player] = []

    def add_player(self, player: Player) -> None:
        """Añade un jugador al equipo"""
        if player.team_id == self.team:
            self.players.append(player)
        else:
            raise ValueError(f"Player team_id {player.team_id} doesn't match team id {self.team}")
    
    def get_players_sorted_by_speed(self, descending: bool = True) -> List[Player]:
        """Obtiene jugadores ordenados por velocidad"""
        valid_players = [p for p in self.players if p.speed is not None]
        return sorted(valid_players, key=lambda p: p.speed, reverse=descending)
    
    def get_players_sorted_by_distance(self, descending: bool = True) -> List[Player]:
        """Obtiene jugadores ordenados por distancia"""
        valid_players = [p for p in self.players if p.distance is not None]
        return sorted(valid_players, key=lambda p: p.distance, reverse=descending)
    
    def get_team_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del equipo"""
        valid_speeds = [p.speed for p in self.players if p.speed is not None]
        valid_distances = [p.distance for p in self.players if p.distance is not None]
        
        return {
            'team_id': self.team,
            'team_name': self.team_name,
            'total_players': len(self.players),
            'avg_speed': np.mean(valid_speeds) if valid_speeds else 0,
            'max_speed': max(valid_speeds) if valid_speeds else 0,
            'avg_distance': np.mean(valid_distances) if valid_distances else 0,
            'max_distance': max(valid_distances) if valid_distances else 0
        }
    
    def clear_players(self) -> None:
        """Limpia la lista de jugadores"""
        self.players.clear()