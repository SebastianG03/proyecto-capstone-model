import numpy as np

def check_speed_consistency(tracks):
    results = {"players": 0, "referees": 0}
    
    for obj in ["players", "referees"]:
        track_speeds = {}
        
        for frame_idx, frame_data in enumerate(tracks[obj]):
            for track_id, track_info in frame_data.items():
                if "speed" in track_info:
                    if track_id not in track_speeds:
                        track_speeds[track_id] = []
                    track_speeds[track_id].append(track_info["speed"])
        
        for track_id, speeds in track_speeds.items():
            if len(speeds) < 2:
                continue
                
            accelerations = np.abs(np.diff(speeds))
            if np.any(accelerations > 15):  
                results[obj] += 1
                
    return results