from app.layers.domain.metrics.imetrics import IMetrics


class Metrics(IMetrics):
    def __init__(self):
        self.processing_time = 0.0
        self.memory_usage = 0.0
        self.interpolation_error = 0.0
        self.ball_detection = {'detected': 0, 'interpolated': 0}
        self.velocity_inconsistencies = {'players': 0, 'referees': 0}
    
    def update_ball_detection(self, detected: int, interpolated: int):
        self.ball_detection['detected'] += detected
        self.ball_detection['interpolated'] += interpolated

    def update_velocity_inconsistencies(self, players: int, referees: int):
        self.velocity_inconsistencies['players'] += players
        self.velocity_inconsistencies['referees'] += referees