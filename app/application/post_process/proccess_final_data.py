from collections import defaultdict
import math
from typing import List, Dict, Optional
from app.core.config import PUBLIC_URL
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
from datetime import datetime
from app.entities.utils.tools_context import AnalysisContext
from app.logger import debug_logger, info_logger


def analyze_match(
    player_states: List[PlayerState],
    ball_events: List[BallEventModel],
    start_time: datetime,
    players: List[Player],
    match_id: int = 0,
) -> List[Dict]:
    """
    Analiza los datos de un partido y devuelve estadísticas por jugador.
    """
    heatmaps_collection = AnalysisContext().tools.heatmap_points
    # Agrupar estados por jugador
    states_by_player = defaultdict(list)
    for state in player_states:
        states_by_player[int(f"{state.player_id}")].append(state)

    # Ordenar por frame_index para cada jugador
    for pid in states_by_player:
        states_by_player[pid].sort(key=lambda s: s.frame_index)

    # Info de jugadores
    player_info = {int(f"{p.player_id}"): p.to_dict() for p in players}

    # Inicializar estadísticas
    stats = {}
    for pid in states_by_player:
        if pid not in player_info:
            continue

        states = states_by_player[pid]

        # Calcular distancia total (acumulada de cada frame)
        total_distance_m = sum(s.distance for s in states if s.distance is not None)
        debug_logger.debug(
            f"[ANALYZE_MATCH] Jugador {pid} - Distancia total (m): {total_distance_m}"
        )
        total_distance_km = total_distance_m / 1000.0

        # Calcular tiempo total considerando todos los frames del jugador
        if len(states) >= 2:
            # Usar el tiempo entre el primer y último frame
            time_s = (states[-1].timestamp_ms - states[0].timestamp_ms) / 1000.0
            # Validar tiempo máximo: 20 minutos = 1200 segundos
            debug_logger.debug(
                f"[ANALYZE_MATCH] Jugador {pid} - Tiempo total (s): {time_s}"
            )
            time_s = min(time_s, 1200.0)
        else:
            time_s = 1.0

        # Calcular velocidad promedio correctamente
        avg_speed_kmh = (total_distance_km / (time_s / 3600.0)) if time_s > 0 else 0.0
        debug_logger.debug(
            f"[ANALYZE_MATCH] Jugador {pid} - Velocidad promedio (km/h): {avg_speed_kmh}"
        )
        # Validar velocidad máxima razonable para un humano (30 km/h)
        avg_speed_kmh = min(avg_speed_kmh, 30.0)

        # Calcular tiempo de posesión total y promedio
        total_possession_time_ms = sum(s.ball_possession_time or 0.0 for s in states)
        debug_logger.debug(
            f"[ANALYZE_MATCH] Jugador {pid} - "
            f"Tiempo total de posesión (ms): {total_possession_time_ms}"
        )
        total_possession_time_s = total_possession_time_ms / 1000.0

        # Calcular promedio de posesión por aparición
        appearances = sum(
            1 for s in states if s.ball_possession_time and s.ball_possession_time > 0
        )
        debug_logger.debug(
            f"[ANALYZE_MATCH] Jugador {pid} - Apariciones: {appearances}"
        )
        avg_possession_time_s = (
            (total_possession_time_s / appearances) if appearances > 0 else 0.0
        )

        # Validar valores máximos
        avg_possession_time_s = min(avg_possession_time_s, 1200.0)
        total_distance_km = min(total_distance_km, 5.0)

        # Obtener datos del jugador con validación
        shirt_number = player_info[pid].get("shirt_number")
        team = player_info[pid].get("team")
        team_color = player_info[pid].get("color")

        # Validar y asignar valores por defecto si es necesario
        if shirt_number is None:
            shirt_number = 0

        if team is None:
            team = "0"

        if team_color is None:
            team_color = "[0, 0, 0]"

        # Redondear hacia arriba cuando corresponda
        def round_up(value, decimals):
            factor = 10**decimals
            return math.ceil(value * factor) / factor

        heatmap = heatmaps_collection.get_by_player_id(pid)
        heatmap_name = f'{heatmap.path}' if heatmap else None
        heatmap_route = f"{PUBLIC_URL}/{heatmap_name.strip()}" if heatmap_name else "https://pub-b1446258d30c4547a877d83f55960843.r2.dev/27/20260118211428_heatmap_player_1_team_None.png"
        info_logger.info(
            f"[ANALYZE_MATCH] Jugador {pid} - Heatmap: {heatmap_route}"
        )

        stats[pid] = {
            "player_id": int(pid),
            "match_id": match_id,
            "shirt_number": shirt_number,
            "team": str(team),
            "team_color": str(team_color),
            "passes": 0,
            "avg_possession_time_s": round_up(avg_possession_time_s, 2),
            "avg_speed_kmh": round_up(avg_speed_kmh, 2),
            "distance_km": round_up(total_distance_km, 3),
            "km_run": round_up(total_distance_km, 3),
            "shots_on_target": 0,
            "has_goal": False,
            "goals": player_info[pid].get("goals"),
            "heatmap_image_path": heatmap_route,
            "started_at": start_time.isoformat(),
        }

    # Contar pases (solo pases, no tiros)
    prev_owner = None
    for event in ball_events:
        item = event.to_dict()
        current_owner = item.get("owner_id")
        if prev_owner and current_owner and prev_owner != current_owner:
            if prev_owner in stats:
                stats[prev_owner]["passes"] += 1
        prev_owner = current_owner

    return list(stats.values())
