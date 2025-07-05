import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        # Court dimensions in meters (width and length)
        COURT_WIDTH = 68
        COURT_LENGTH = 23.32

        # Define source quadrilateral in image pixels
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner
            [265, 275],   # Top-left corner
            [910, 260],   # Top-right corner
            [1640, 915]   # Bottom-right corner
        ], dtype=np.float32)

        # Define target rectangle in real-world coordinates (meters)
        self.target_vertices = np.array([
            [0, COURT_WIDTH],         # Bottom-left
            [0, 0],                    # Top-left
            [COURT_LENGTH, 0],         # Top-right
            [COURT_LENGTH, COURT_WIDTH] # Bottom-right
        ], dtype=np.float32)

        # Create perspective transformation matrix
        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, 
            self.target_vertices
        )

    def transform_point(self, point):
        """Transform a point from image coordinates to court coordinates"""
        # Convert to integer for polygon test
        int_point = (int(point[0]), int(point[1]))
        
        # Check if point is within the court boundaries
        if cv2.pointPolygonTest(self.pixel_vertices, int_point, False) < 0:
            return None

        # Reshape point to OpenCV format: (1, 1, 2)
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, 
            self.perspective_transform
        )
        
        # Return as flat (x,y) coordinates
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """Add transformed positions to tracking data"""
        for object_type, object_tracks in tracks.items():
            for frame_idx, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    # Get adjusted position from tracking data
                    position_adjusted = np.array(track_info['position_adjusted'])
                    
                    # Transform to court coordinates
                    position_transformed = self.transform_point(position_adjusted)
                    
                    # Store result (converted to list if valid)
                    track_info['position_transformed'] = (
                        position_transformed.squeeze().tolist() 
                        if position_transformed is not None 
                        else None
                    )