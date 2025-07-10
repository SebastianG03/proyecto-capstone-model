import time
import tracemalloc
import numpy as np
from cv2.typing import MatLike
from layers.infraestructure.validation.interpolation_validation import calculate_interpolation_error
from layers.infraestructure.validation.system_usage_validation import start_memory_usage
from layers.infraestructure.validation.velocity_consistence import check_speed_consistency
from layers.presentation.diagram_processor import generate_diagrams
from layers.presentation.tracker_initiation import init_tracker
from layers.infraestructure.video_analysis.trackers.tracker import Tracker

from layers.infraestructure.video_analysis.services.video_processing_service import (
    read_video, save_video)
from layers.infraestructure.video_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from layers.infraestructure.video_analysis.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from layers.infraestructure.video_analysis.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from layers.infraestructure.video_analysis.team_assigner.team_assigner import TeamAssigner
from layers.infraestructure.video_analysis.view_transformer.view_transformer import ViewTransformer

def main():
    # Initialize metrics and performance tracking
    tracemalloc.start()  # Start memory tracking
    start_time = time.time()
    metrics = {
        'processing_time': [],
        'memory_usage': [],
        'ball_detection': {'detected': 0, 'interpolated': 0},
        'interpolation_error': 0.0,
        'velocity_inconsistencies': {'players': 0, 'referees': 0}
    }
    
    # Read Video
    video_frames = read_video('./res/input_videos/08fd33_4.mp4')
    if not video_frames:
        print("Error: No frames read from video")
        return

    # Initialize components
    tracker = init_tracker(11, 'n', './res/models/yolo')
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # Process video in bulk (not per-frame)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='./res/stubs/track_stubs.pkl'
    )
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Estimate camera movement
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='./res/stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformation
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    original_ball_tracks = tracks["ball"].copy()
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Count ball detections (efficiently)
    detected_frames = sum(1 for frame in original_ball_tracks if 1 in frame)
    metrics['ball_detection'] = {
        'detected': detected_frames,
        'interpolated': len(original_ball_tracks) - detected_frames
    }

    # Speed and distance estimation
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],   
                track['bbox'],
                player_id
            )
            track['team'] = team 
            track['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player_track[assigned_player]['has_ball'] = True
            team_ball_control.append(player_track[assigned_player]['team'])
        else:
            # Handle first frame case
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
    
    team_ball_control = np.array(team_ball_control)

    # Calculate metrics
    metrics['interpolation_error'] = calculate_interpolation_error(
        tracker,  # Pass tracker instance
        original_ball_tracks
    )
    metrics['velocity_inconsistencies'] = check_speed_consistency(tracks)
    
    # Draw output 
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, './res/output_videos/output_video.avi')

    # Generate diagrams (will save each metric separately)
    generate_diagrams(tracks=tracks, metrics=metrics)

    # Final metrics report
    total_time = time.time() - start_time
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    total_memory = sum(stat.size for stat in top_stats) / (1024 * 1024)  # MB

    # Ensure we have memory measurements
    if not metrics['memory_usage']:
        # Add final memory measurement if no others exist
        metrics['memory_usage'].append(total_memory)

    print("\n" + "="*50)
    print("RESUMEN DE MÉTRICAS DE RENDIMIENTO")
    print("="*50)
    print(f"Tiempo total de procesamiento: {total_time:.2f} s")
    print(f"Tiempo promedio por frame: {total_time / len(video_frames):.4f} s")
    print(f"Uso máximo de memoria: {max(metrics['memory_usage']):.2f} MB")
    print(f"Detección de balón: {metrics['ball_detection']['detected']} frames "
          f"({metrics['ball_detection']['detected']/len(original_ball_tracks)*100:.1f}%)")
    print(f"Inconsistencias de velocidad: Jugadores={metrics['velocity_inconsistencies']['players']}, "
          f"Árbitros={metrics['velocity_inconsistencies']['referees']}")
    print(f"Error de interpolación: {metrics['interpolation_error']:.4f}")

if __name__ == '__main__':
    main()