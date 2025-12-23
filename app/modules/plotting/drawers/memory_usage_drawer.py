from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram
from matplotlib import pyplot as plt


class MemoryUsageDrawer(Diagram):
    def __init__(self, tracks: dict, metrics: dict):
        super().__init__(tracks, metrics)
        self.save_path = '../app/res/output_videos/memory_usage.png'

    def draw_and_save(self) -> None:
        if self.metrics is None or 'memory_usage' not in self.metrics:
            print("No memory usage data available to plot.")
            return
        memory = self.metrics['memory_usage']
        plt.figure(figsize=(10, 6))
        plt.plot(memory, 'g-')
        plt.title('Uso de Memoria durante la Ejecución')
        plt.xlabel('Punto de Medición')
        plt.ylabel('Memoria (MB)')
        plt.grid(True)
        plt.savefig(self.save_path, dpi=120)
        plt.close()
