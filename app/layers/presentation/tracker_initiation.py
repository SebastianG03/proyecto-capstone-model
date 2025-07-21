from layers.infraestructure.video_analysis.trackers.tracker import Tracker


def init_tracker(version: int, size: str, save_path: str) -> Tracker:
    return Tracker(f'{save_path}{version}{size}.pt')