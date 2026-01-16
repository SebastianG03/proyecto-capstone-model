from collections import defaultdict
from typing import List, Dict, Optional
from app.core.config import PUBLIC_URL
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
from datetime import datetime
from decouple import config


def analyze_match(
    player_states: List[PlayerState],
    ball_events: List[BallEventModel],
    start_time: datetime,
    players: List[Player],
    heatmaps: Optional[Dict[int, str]] = None,
    match_id: int = 0,
) -> List[Dict]:
    """
    Analiza los datos de un partido y devuelve estadísticas por jugador.
    """
    # Agrupar estados por jugador
    states_by_player = defaultdict(list)
    for state in player_states:
        states_by_player[state.player_id].append(state)

    # Ordenar por frame_index
    for pid in states_by_player:
        states_by_player[pid].sort(key=lambda s: s.frame_index)

    # Info de jugadores
    player_info = {p.player_id: p.to_dict() for p in players}

    # Inicializar estadísticas
    stats = {}
    for pid in states_by_player:
        if pid not in player_info:
            continue

        states = states_by_player[pid]
        total_distance_m = sum(s.distance for s in states if s.distance is not None)
        total_distance_km = total_distance_m / 1000.0

        # Tiempo total en segundos
        timestamps = [s.timestamp_ms for s in states if s.timestamp_ms is not None]
        if len(timestamps) >= 2:
            time_s = (max(timestamps) - min(timestamps)) / 1000.0
        else:
            time_s = 1.0  # fallback para evitar división por cero

        # Velocidad promedio
        avg_speed_kmh = (total_distance_km / (time_s / 3600.0)) if time_s > 0 else 0.0

        # Tiempo de posesión
        total_possession_time = sum(s.ball_possession_time or 0.0 for s in states)

        stats[pid] = {
            "player_id": int(pid),
            "match_id": match_id,
            "shirt_number": player_info[pid]["shirt_number"],
            "team": str(player_info[pid]["team"]),
            "team_color": str(player_info[pid]["color"]),
            "passes": 0,
            "avg_possession_time_s": total_possession_time,
            "avg_speed_kmh": avg_speed_kmh,
            "distance_km": total_distance_km,
            "km_run": total_distance_km,
            "shots_on_target": 0,
            "has_goal": False,
            "heatmap_image_path": "",
            "started_at": start_time.isoformat()
        }

    # Contar pases y tiros (lógica original)
    prev_owner = None
    for event in ball_events:
        item = event.to_dict()
        current_owner = item.get("owner_id")
        if prev_owner and current_owner and prev_owner != current_owner:
            if prev_owner in stats:
                stats[prev_owner]["passes"] += 1
                stats[prev_owner]["shots_on_target"] += 1  # ¿Lógica correcta?
        prev_owner = current_owner

    # Agregar heatmaps
    if heatmaps:
        for pid, key in heatmaps.items():
            if pid in stats:
                stats[pid]["heatmap_image_path"] = f"{PUBLIC_URL}/{key}"

    return list(stats.values())