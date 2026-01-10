from abc import ABC, abstractmethod
from typing import Dict
from sqlalchemy.orm import Session



class Diagram(ABC):
    def __init__(self, db: Session, *args, **kwargs):
        self.db = db

    @abstractmethod
    def draw_and_save(self) -> None:
        pass
