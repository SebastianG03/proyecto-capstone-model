from typing import List, Dict

class GoalTeamResolver:
    def __init__(self, goal_detector, track_manager):
        """
        - goal_detector: instancia de tu detector de goles (lógica de frames)
        - track_manager: clase/objeto que mantiene estado de tracking
        """
        self.goal_detector = goal_detector
        self.track_manager = track_manager

    def process_frame(self, detections: List[Dict]):
        """
        Devuelve:
          {
            "goal": bool,
            "scoring_team": str or None,
            "scoring_player_id": int or None
          }
        """
        # Primero actualizamos detector de gol
        is_goal = self.goal_detector.update(detections)

        if not is_goal:
            return {"goal": False, "scoring_team": None, "scoring_player_id": None}

        # Si hay gol, sacamos el último ID de jugador que tocó balón
        last_touch_id = self.track_manager.get_last_ball_touch_player_id()

        if last_touch_id is None:
            return {"goal": True, "scoring_team": None, "scoring_player_id": None}

        # Y ahora mapear ID → equipo
        team = self.track_manager.get_team_by_player_id(last_touch_id)

        return {
            "goal": True,
            "scoring_team": team,
            "scoring_player_id": last_touch_id
        }
