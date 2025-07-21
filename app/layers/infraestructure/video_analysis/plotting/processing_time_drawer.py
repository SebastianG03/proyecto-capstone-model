# layers/infraestructure/video_analysis/plotting/metrics_dashboard_drawer.py
import matplotlib.pyplot as plt
import numpy as np
from .diagram import Diagram

class ProcessingTimeDrawer(Diagram):
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.save_path = '../app/res/output_videos/processing_time.png'
    
    def draw_and_save(self) -> None:
        times = self.metrics['processing_time']
        frames = range(1, len(times) + 1)
        cumulative_avg = [np.mean(times[:i]) for i in frames]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, times, 'o-', alpha=0.5, label='Tiempo por Frame')
        plt.plot(frames, cumulative_avg, 'r-', label='Promedio Acumulado')
        plt.title('Tiempo de Procesamiento por Frame')
        plt.xlabel('Número de Frame')
        plt.ylabel('Tiempo (s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path, dpi=120)
        plt.close()


