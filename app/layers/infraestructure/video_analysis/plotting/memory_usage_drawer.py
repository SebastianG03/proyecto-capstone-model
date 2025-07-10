from matplotlib import pyplot as plt

from layers.infraestructure.video_analysis.plotting.diagram import Diagram


class MemoryUsageDrawer(Diagram):
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.save_path = '../app/res/output_videos/memory_usage.png'
    
    def draw_and_save(self) -> None:
        memory = self.metrics['memory_usage']
        plt.figure(figsize=(10, 6))
        plt.plot(memory, 'g-')
        plt.title('Uso de Memoria durante la Ejecución')
        plt.xlabel('Punto de Medición')
        plt.ylabel('Memoria (MB)')
        plt.grid(True)
        plt.savefig(self.save_path, dpi=120)
        plt.close()