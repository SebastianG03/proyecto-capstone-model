from typing import List
from layers.infraestructure.video_analysis.plotting.voronoi_diagram_drawer import VoronoiDiagramDrawer


class DrawerFactory:
    @staticmethod
    def run_drawer(drawer_type: str, players_tracks: List) -> None:
        diagram_map = {
            'voronoi': VoronoiDiagramDrawer,
        }


        diagram = diagram_map.get(drawer_type.lower())
        if not diagram:
            raise ValueError(f"Drawer type '{drawer_type}' is not supported.")
        
        diagram_instance = diagram(players_tracks)
        diagram_instance.draw_and_save()