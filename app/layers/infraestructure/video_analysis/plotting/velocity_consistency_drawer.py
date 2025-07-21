from matplotlib import pyplot as plt

from layers.infraestructure.video_analysis.plotting.diagram import Diagram


class VelocityConsistencyDrawer(Diagram):
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.save_path = '../app/res/output_videos/velocity_consistency.png'
    
    def draw_and_save(self) -> None:
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