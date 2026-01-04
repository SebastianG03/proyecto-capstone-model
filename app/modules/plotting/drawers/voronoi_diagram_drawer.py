from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from app.modules.plotting.interfaces import \
    Diagram
from app.modules.plotting.services.drawer_service import \
    DrawerService
from mplsoccer import Pitch


class VoronoiDiagramDrawer(Diagram):
    def __init__(self, tracks: Dict):  # Cambiado a lista de frames
        self.home_team_color = 'blue'
        self.rival_team_color = 'red'
        self.save_path = './app/res/output_videos/voronoi_diagram.png'
        self.drawer_service = DrawerService()

    def draw_and_save(self) -> None:
        pass