from typing import Dict
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram
from matplotlib import pyplot as plt


class InterpolationErrorDrawer(Diagram):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None):
        super().__init__(tracks)
        self.save_path = './app/res/output_videos/interpolation_error.png'

    def draw_and_save(self) -> None:
        if not self.metrics or 'interpolation_error' not in self.metrics:
            return
        error = self.metrics['interpolation_error']
        plt.figure(figsize=(6, 6))
        plt.bar(['Error MSE'], [error], color='#FF5722')
        plt.title('Error de Interpolación del Balón', fontweight='bold')
        plt.ylabel('Error Cuadrático Medio')
        plt.grid(axis='y')
        plt.text(0, error + 0.01, f"{error:.4f}",
                 ha='center', fontweight='bold')
        plt.savefig(self.save_path, dpi=120)
        plt.close()
