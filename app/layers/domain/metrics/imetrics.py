from abc import ABC


class IMetrics(ABC):
    processing_time: float
    memory_usage: float
    ball_detection: dict = {'detected': 0, 'interpolated': 0}
    interpolation_error: float
    velocity_inconsistencies: dict = {'players': 0, 'referees': 0}