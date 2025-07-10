from typing import List
from layers.infraestructure.video_analysis.plotting.drawer_factory import DrawerFactory


def generate_diagrams(tracks: List, metrics: List) -> None:
    try:
        DrawerFactory.run_drawer('voronoi', tracks['players'])
        DrawerFactory.run_drawer('heatmap', tracks['players'])
        DrawerFactory.run_drawer('ball_drawer', tracks['players'])
        DrawerFactory.run_drawer('processing_time', metrics)
        DrawerFactory.run_drawer('memory_usage', metrics)
        DrawerFactory.run_drawer('ball_detection', metrics)
        DrawerFactory.run_drawer('velocity_consistency', metrics)
        DrawerFactory.run_drawer('interpolation_error', metrics)
    except Exception as e:
        print(f"Error drawing diagram: {e}")