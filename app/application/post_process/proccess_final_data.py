from collections import defaultdict
from typing import List, Dict, Optional
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
import random

BASE_URL = "https://pub-b1446258d30c4547a877d83f55960843.r2.dev"

unique_shirt_numbers = set(
    range(1, 100)
)

def analyze_match(
    player_states: List[PlayerState],
    ball_events: List[BallEventModel],
    players: List[Player],
    heatmaps: Optional[Dict[int, str]] = None,
    match_id: int = 0
) -> List[Dict]:
    """
    Analiza los datos de un partido y devuelve estadísticas por jugador en formato compatible con el serializer.
    """
    stats = defaultdict(lambda: {
        "player_id": 0,
        "match_id": match_id,
        "shirt_number": None,
        "team": None,
        "team_color": None,
        "passes": 0,
        "avg_possession_time_s": 0.0,
        "avg_speed_kmh": 0.0,
        "distance_km": 0.0,
        "km_run": None,
        "shots_on_target": 0,
        "has_goal": False,
        "heatmap_image_path": ""
    })

    player_info = {p.player_id: p.to_dict() for p in players}

    for record in player_states:
        item = record.to_dict()
        pid = record.player_id
        if pid not in player_info:
            continue

        s = stats[pid]
        s["player_id"] = int(f'{pid}')
        s["match_id"] = match_id
        
        s["shirt_number"] = player_info[pid]["shirt_number"]
        s["team"] = str(player_info[pid]["team"])
        s["team_color"] = str(player_info[pid]["color"])

        # Acumulados
        s["avg_possession_time_s"] += float(item.get("ball_possession_time", 0.0))
        s["distance_km"] += float(item.get("incremental_distance", 0.0)) / 1000.0  # m -> km
        s["avg_speed_kmh"] += float(item.get("speed", 0.0))

    # Promedios
    total_frames = len(player_states)
    for pid in stats:
        if total_frames > 0:
            stats[pid]["avg_speed_kmh"] /= total_frames

    # Pases y tiros
    prev_owner = None
    for event in ball_events:
        item = event.to_dict()
        current_owner = item.get("ball_owner_id")
        if prev_owner and current_owner and prev_owner != current_owner:
            stats[prev_owner]["passes"] += 1
            stats[prev_owner]["shots_on_target"] += 1
        prev_owner = current_owner

    # Agregar heatmap si existe
    if heatmaps:
        for pid, key in heatmaps.items():
            if pid in stats:
                stats[pid]["heatmap_image_path"] = f"{BASE_URL}/{key}"

    # Convertir a lista y aplicar formato final
    result = []
    for pid, data in stats.items():
        data["km_run"] = data["distance_km"]  # alias legacy
        result.append(data)

    return result