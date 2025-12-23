from typing import Dict
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram
from matplotlib import pyplot as plt


class BallDetectionMetricsDrawer(Diagram):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None):
        super().__init__(tracks)
        self.save_path = './app/res/output_videos/ball_detection.png'

    def draw_and_save(self) -> None:
        if not self.metrics or 'ball_detection' not in self.metrics:
            return
        detection = self.metrics['ball_detection']
        labels = ['Detectados', 'Interpolados']
        sizes = [detection['detected'], detection['interpolated']]

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                colors=['#4CAF50', '#FFC107'], startangle=90)
        plt.title('Detección de Balón')
        plt.axis('equal')
        plt.savefig(self.save_path, dpi=120)
        plt.close()
