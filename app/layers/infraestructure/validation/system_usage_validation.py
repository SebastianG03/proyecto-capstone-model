import tracemalloc
def calculate_frame_processing_time(final: float, start: float, video_frames: list) -> float:
    return (final - start) / len(video_frames)

def start_memory_usage() -> float:
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')