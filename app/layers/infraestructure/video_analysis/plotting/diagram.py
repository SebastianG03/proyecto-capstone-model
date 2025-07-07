from abc import ABC, abstractmethod
from typing import List, Dict, Any
from layers.infraestructure.video_analysis.plotting.drawer_service import DrawerService


class Diagram(ABC):
    @abstractmethod
    def draw_and_save(self) -> None:
        pass