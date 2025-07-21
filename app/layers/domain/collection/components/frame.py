from dataclasses import field
from typing import Any, Dict, List, Optional
from app.layers.domain.collection.components.team import Team
from app.layers.domain.player.player import Player


class Frame():
    frame: int
    teams: list[Team] = []
    timestamp: float = field(default_factory=lambda: 0.0)
    

    def add_team(self, team: Team) -> None:
        """Añade un equipo al frame"""
        self.teams.append(team)
    
    def get_team_by_id(self, team_id: int) -> Optional[Team]:
        """Obtiene un equipo por su ID"""
        for team in self.teams:
            if team.team == team_id:
                return team
        return None
    
    def get_all_players(self) -> List[Player]:
        """Obtiene todos los jugadores del frame"""
        all_players = []
        for team in self.teams:
            all_players.extend(team.players)
        return all_players
    
    def get_all_players_sorted_by_speed(self, descending: bool = True) -> List[Player]:
        """Obtiene todos los jugadores ordenados por velocidad"""
        all_players = self.get_all_players()
        valid_players = [p for p in all_players if p.speed is not None]
        return sorted(valid_players, key=lambda p: p.speed, reverse=descending)
    
    def get_frame_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del frame"""
        all_players = self.get_all_players()
        valid_speeds = [p.speed for p in all_players if p.speed is not None]
        valid_distances = [p.distance for p in all_players if p.distance is not None]
        
        team_stats = [team.get_team_stats() for team in self.teams]
        
        return {
            'frame_number': self.frame,
            'timestamp': self.timestamp,
            'total_teams': len(self.teams),
            'total_players': len(all_players),
            'avg_speed': np.mean(valid_speeds) if valid_speeds else 0,
            'max_speed': max(valid_speeds) if valid_speeds else 0,
            'team_stats': team_stats
        }