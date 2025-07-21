import numpy as np

def calculate_interpolation_error(tracker, original_tracks):
    interpolated_tracks = tracker.interpolate_ball_positions(original_tracks.copy())
    errors = []
    
    for orig_frame, interp_frame in zip(original_tracks, interpolated_tracks):
        if 1 in orig_frame and 1 in interp_frame:
            orig_bbox = orig_frame[1]["bbox"]
            interp_bbox = interp_frame[1]["bbox"]
            
            orig_center = np.array([
                (orig_bbox[0] + orig_bbox[2]) / 2,
                (orig_bbox[1] + orig_bbox[3]) / 2
            ])
            
            interp_center = np.array([
                (interp_bbox[0] + interp_bbox[2]) / 2,
                (interp_bbox[1] + interp_bbox[3]) / 2
            ])
            
            error = np.sum((orig_center - interp_center) ** 2)
            errors.append(error)
    
    return np.mean(errors) if errors else 0.0