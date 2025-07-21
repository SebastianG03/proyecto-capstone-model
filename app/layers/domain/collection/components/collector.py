from dataclasses import field
from app.layers.domain.collection.components.frame import Frame

class Collector():
    id: int
    frames: list[Frame] = field(default_factory=list)