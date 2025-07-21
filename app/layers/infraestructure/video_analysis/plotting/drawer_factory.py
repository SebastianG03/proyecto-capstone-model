from typing import List
from layers.infraestructure.video_analysis.plotting.ball_detection_metrics_drawer import BallDetectionMetricsDrawer
from layers.infraestructure.video_analysis.plotting.interpolation_error_drawer import InterpolationErrorDrawer
from layers.infraestructure.video_analysis.plotting.memory_usage_drawer import MemoryUsageDrawer
from layers.infraestructure.video_analysis.plotting.processing_time_drawer import ProcessingTimeDrawer
from layers.infraestructure.video_analysis.plotting.velocity_consistency_drawer import VelocityConsistencyDrawer
from layers.infraestructure.video_analysis.plotting.ball_trajectory_drawer import BallTrajectoryDrawer
from layers.infraestructure.video_analysis.plotting.heatmap_drawer import HeatmapDrawer
from layers.infraestructure.video_analysis.plotting.voronoi_diagram_drawer import VoronoiDiagramDrawer


class DrawerFactory:
    @staticmethod
    def run_drawer(drawer_type: str, players_tracks: List) -> None:
        diagram_map = {
            'voronoi': VoronoiDiagramDrawer,
            'heatmap': HeatmapDrawer,
            'ball_drawer': BallTrajectoryDrawer,
            'processing_time': ProcessingTimeDrawer,
            'memory_usage': MemoryUsageDrawer,
            'ball_detection': BallDetectionMetricsDrawer,
            'velocity_consistency': VelocityConsistencyDrawer,
            'interpolation_error': InterpolationErrorDrawer
        }


        diagram = diagram_map.get(drawer_type.lower())
        if not diagram:
            raise ValueError(f"Drawer type '{drawer_type}' is not supported.")
        
        diagram_instance = diagram(players_tracks)
        diagram_instance.draw_and_save()