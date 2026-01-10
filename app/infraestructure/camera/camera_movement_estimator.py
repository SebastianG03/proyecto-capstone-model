import cv2
import numpy as np
from cv2.typing import MatLike

from app.entities.models import PlayerState, BallEventModel
from app.entities.utils import Singleton
from app.logger import debug_logger, error_logger
from sqlalchemy.orm import Session


class CameraMovementEstimator(metaclass=Singleton):
    """
    Versión STREAMING del estimador de movimiento de cámara.
    Mantiene el nombre de la clase original.

    USO:
        estimator = CameraMovementEstimator(first_frame)
        movement = estimator.update(frame_t)
    """

    def __init__(self, first_frame: MatLike):
        self.minimum_distance = 5
        self.accum_scale = 1.0
        self.last_scale = 1.0
        self.accum_dx = 0.0
        self.accum_dy = 0.0

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

        # Estado interno
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        mask_features = np.zeros_like(self.old_gray)
        mask_features[:, 0:20] = 1
        mask_features[:, -150:] = 1

        self.features_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

        self.old_features = cv2.goodFeaturesToTrack(
            self.old_gray,
            **self.features_params, # type: ignore
            
        )

        # Último movimiento estimado → smoothing
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.alpha = 0.35  # smoothing EMA

    # -------------------------------------------------------------
    # STREAMING UPDATE
    # -------------------------------------------------------------
    def update(self, frame: MatLike):
        """
        Procesa UN SOLO FRAME y retorna el movimiento:
        (dx, dy)

        dx > 0 → cámara se mueve hacia la derecha
        dy > 0 → cámara se mueve hacia abajo
        """

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_features is None or len(self.old_features) == 0:
            self.old_features = cv2.goodFeaturesToTrack(
                frame_gray, **self.features_params # type: ignore
            )

            self.old_gray = frame_gray
            return 0.0, 0.0

        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray,
            frame_gray,
            self.old_features,
            None, # type: ignore
            **self.lk_params # type: ignore
        )

        dx, dy, dist, scale = self.update_camera_distance(new_features, self.old_features)

        # Threshold para considerar movimiento real
        if dist > self.minimum_distance:
            self.old_features = cv2.goodFeaturesToTrack(
                frame_gray, **self.features_params # type: ignore
            )

        # Smoothing (EMA)
        scale_smooth = (self.alpha * scale) + ((1 - self.alpha) * self.last_scale)
        self.last_scale = scale_smooth
        dx_smooth = (self.alpha * dx) + ((1 - self.alpha) * self.last_dx)
        dy_smooth = (self.alpha * dy) + ((1 - self.alpha) * self.last_dy)

        self.accum_dx += dx_smooth
        self.accum_dy += dy_smooth
        self.accum_scale *= scale_smooth

        self.last_dx, self.last_dy = dx_smooth, dy_smooth
        self.old_gray = frame_gray.copy()

        return dx_smooth, dy_smooth

    def get_current_scale(self) -> float:
        """
        Returns the current scale of the camera movement estimation.

        The current scale is calculated as the exponential moving average (EMA)
        of the scale of the camera movement. This is useful for determining the
        zoom level of the video.

        Returns
        -------
        float
            The current scale of the camera movement estimation
        """
        return self.accum_scale

    # -------------------------------------------------------------
    # APLICAR AJUSTE A TRACK
    # -------------------------------------------------------------
    def add_adjust_positions_to_tracks(
        self,
        camera_movement_per_frame,
        scale: float,
        pixels_to_meters: float,
        track: PlayerState | BallEventModel,
        db: Session
    ):
        """
        Ajusta la posición del jugador/ balón compensando movimiento de cámara.
        """
        try:
            tracks_collection = None
            print(f"Ajustando posición del track {track.id} con movimiento de cámara {camera_movement_per_frame}.")
            dx, dy = camera_movement_per_frame
            debug_logger.debug(f"[CameraMovementEstimator] Movimiento de cámara por frame: dx={dx}, dy={dy}")

            if dx is None or dy is None:
                print("Movimiento de cámara no definido, no se aplica ajuste.")
                return

            #Conversion
            # dx *= pixels_to_meters
            # dy *= pixels_to_meters
            # debug_logger.debug(f"[CameraMovementEstimator] Movimiento de cámara convertido a metros: dx={dx}, dy={dy}")
            debug_logger.debug(f"[CameraMovementEstimator] Movimiento de cámara: dx={dx}, dy={dy}")
            x, y = track.x, track.y

            if x is None or y is None:
                print("Posición del track no definida, no se aplica ajuste.")
                return

            #Conversion
            debug_logger.debug(f"[CameraMovementEstimator] Posición actual: x={x}, y={y}")
            # x *= pixels_to_meters
            # y *= pixels_to_meters
            # debug_logger.debug(f"[CameraMovementEstimator] Posición del track convertida a metros: x={x}, y={y}")

            adjusted_x = (x - dx) / scale
            adjusted_y = (y - dy) / scale
            position_adjusted = (adjusted_x, adjusted_y)

            if position_adjusted[0] is None or position_adjusted[1] is None:
                debug_logger.debug("Posición ajustada inválida, no se aplica ajuste.")
                return
            debug_logger.debug(f"Posición ajustada: x={position_adjusted[0]}, y={position_adjusted[1]}")

            updates = {
                "x": position_adjusted[0],
                "y": position_adjusted[1]
            }
            debug_logger.debug(f"Actualizaciones a aplicar: {updates}")

            debug_logger.debug(f"Actualizando track ID {track.id} en la base de datos.")
            if isinstance(track, PlayerState):
                from app.entities.collections import TrackCollectionPlayer
                debug_logger.debug("Usando TrackCollectionPlayer para actualizar el track.")
                tracks_collection = TrackCollectionPlayer(db)
                tracks_collection.patch_state(
                    int(f'{track.player_id}'),
                    int(f'{track.frame_index}'),
                    updates)
            elif isinstance(track, BallEventModel):
                from app.entities.collections import TrackCollectionBall
                debug_logger.debug("Usando TrackCollectionBall para actualizar el track.")
                tracks_collection = TrackCollectionBall(db)
                tracks_collection.patch(
                    int(f'{track.id}'),
                    updates)
            if not tracks_collection:
                error_logger.error("tracks_collection no pudo ser determinado.")
                raise ValueError("tracks_collection no pudo ser determinado.")

            debug_logger.debug(f"Posición del track {track.id} ajustada correctamente.")

        except Exception as e:
            debug_logger.debug(f"Error ajustando posición del track {track}: {e}")
            raise e

    # -------------------------------------------------------------
    # CALCULAR DISTANCIA ENTRE FEATURES
    # -------------------------------------------------------------
    def update_camera_distance(self, new_features, old_features):
        if new_features is None or old_features is None:
            return 0.0, 0.0, 0.0, 1.0

        if len(new_features) != len(old_features) or len(new_features) == 0:
            return 0.0, 0.0, 0.0, 1.0
        
        deltas = new_features - old_features
        dx =  float(np.median(deltas[:, 0, 0]))
        dy =  float(np.median(deltas[:, 0, 1]))
        
        src = old_features.reshape(-1, 2).astype(np.float32)
        dst = new_features.reshape(-1, 2).astype(np.float32)
        
        M, _ = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3)
        
        scale = np.sqrt(M[0,0]**2 + M[1,0]**2) if M is not None else 1.0
        distance = np.sqrt(dx**2 + dy**2)

        return dx, dy, distance, scale
        # max_distance = 0.0
        # camera_movement_x = 0.0
        # camera_movement_y = 0.0

        # for new_feat, old_feat in zip(new_features, old_features):
        #     new_point = new_feat.ravel()
        #     old_point = old_feat.ravel()

        #     diff = new_point - old_point
        #     distance = np.linalg.norm(diff)

        #     if distance > max_distance:
        #         max_distance = distance
        #         camera_movement_x = float(diff[0])
        #         camera_movement_y = float(diff[1])

        # return camera_movement_x, camera_movement_y, float(max_distance)
