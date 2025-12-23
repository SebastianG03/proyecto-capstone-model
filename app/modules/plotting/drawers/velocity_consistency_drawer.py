from typing import Dict
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram
from matplotlib import pyplot as plt


class VelocityConsistencyDrawer(Diagram):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None):
        super().__init__(tracks)
        self.save_path = './app/res/output_videos/velocity_consistency.png'

    def draw_and_save(self) -> None:
        if not self.metrics or 'velocity_inconsistencies' not in self.metrics:
            return
        inconsistencies = self.metrics['velocity_inconsistencies']
        objects = ['Jugadores', 'Árbitros']
        counts = [inconsistencies['players'], inconsistencies['referees']]

        plt.figure(figsize=(8, 6))
        plt.bar(objects, counts, color=['#2196F3', '#9C27B0'])
        plt.title('Inconsistencias de Velocidad', fontweight='bold')
        plt.ylabel('Conteo')
        plt.grid(axis='y')

        for i, v in enumerate(counts):
            plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        plt.savefig(self.save_path, dpi=120)
        plt.close()
