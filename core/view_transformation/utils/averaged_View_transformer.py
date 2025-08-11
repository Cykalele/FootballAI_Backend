import numpy as np
import cv2

class AveragedViewTransformer:
    """
    Applies a fixed homography to convert image-space detections into metric pitch coordinates,
    with optional temporal smoothing per track ID via a lightweight constant-velocity Kalman filter.

    This is an original implementation for FootballAI and is not adapted from an external repository.
    It combines three main elements:
    1. Homography transformation (OpenCV perspectiveTransform)
    2. Outlier rejection based on maximum displacement per frame
    3. Per-ID Kalman filter smoothing to reduce jitter in tracking

    Parameters
    ----------
    homography_matrix : np.ndarray
        3x3 matrix mapping image coordinates to pitch coordinates.
    distance_threshold : float
        Maximum allowed displacement between consecutive frames (in cm) before treating a point as an outlier.
    dt : float
        Frame time interval in seconds, used for the constant-velocity model.
    """

    def __init__(self, homography_matrix: np.ndarray, distance_threshold: float = 40.0, dt: float = 1/30.0):
        if homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3.")
        self.homography_matrix = homography_matrix.astype(np.float32)
        self.distance_threshold = float(distance_threshold)
        self.dt = float(dt)

        # Store the last accepted position per ID for outlier filtering
        self.last_valid_positions: dict[int, np.ndarray] = {}

        # Per-ID Kalman filter states: player_id -> dict(F, H, Q, R, P, x)
        self.filters: dict[int, dict] = {}

        # Internal identifier for the ball track (treated as single-ID track)
        self._ball_id = 0

    def _init_kf(self, z_xy: np.ndarray) -> dict:
        """
        Initialize a constant-velocity Kalman filter for a new track.

        The filter state vector is [x, y, vx, vy]^T, with position in cm and velocity in cm/s.

        Parameters
        ----------
        z_xy : np.ndarray
            Initial measurement [x, y] in cm.

        Returns
        -------
        dict
            Kalman filter state and matrices.
        """
        dt = self.dt
        # State transition: constant velocity
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=np.float32)
        # Observation model: measure only x and y
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)

        # Process noise (Q) and measurement noise (R) tuned conservatively
        Q = np.diag([5, 5, 50, 50]).astype(np.float32) * dt
        R = np.diag([25, 25]).astype(np.float32)

        # Initial covariance: high uncertainty in initial velocity
        P = np.eye(4, dtype=np.float32) * 100.0
        # Initial state: zero velocity
        x = np.array([z_xy[0], z_xy[1], 0.0, 0.0], dtype=np.float32)

        return {"F": F, "H": H, "Q": Q, "R": R, "P": P, "x": x}

    @staticmethod
    def _kf_update(f: dict, z_xy: np.ndarray) -> np.ndarray:
        """
        Perform a single predict–update cycle of the Kalman filter.

        Parameters
        ----------
        f : dict
            Kalman filter state and matrices.
        z_xy : np.ndarray
            New measurement [x, y] in cm.

        Returns
        -------
        np.ndarray
            Updated position estimate [x, y] after smoothing.
        """
        F, H, Q, R, P, x = f["F"], f["H"], f["Q"], f["R"], f["P"], f["x"]

        # Predict step
        x = F @ x
        P = F @ P @ F.T + Q

        # Innovation (measurement residual)
        y = z_xy - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Update step
        x = x + K @ y
        I = np.eye(4, dtype=np.float32)
        P = (I - K @ H) @ P

        f["x"], f["P"] = x, P
        return x[:2]

    def _transform_raw(self, points: np.ndarray) -> np.ndarray:
        """
        Apply homography transformation to an Nx2 array of points.

        Parameters
        ----------
        points : np.ndarray
            Array of image coordinates (N, 2).

        Returns
        -------
        np.ndarray
            Transformed pitch coordinates (N, 2) in cm.
        """
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.homography_matrix).reshape(-1, 2)
        return transformed.astype(np.float32)

    def transform_points(
        self,
        points: np.ndarray,
        player_ids: list[int] = None
    ) -> dict[int, np.ndarray] | np.ndarray:
        """
        Transform and smooth detected points from image space to pitch space.

        If `player_ids` is provided:
            - Applies outlier rejection per ID.
            - Smooths each track with a Kalman filter.

        If `player_ids` is None:
            - Assumes a single-object track (ball).
            - Applies Kalman smoothing to the single coordinate.

        Parameters
        ----------
        points : np.ndarray
            Array of image coordinates (N, 2).
        player_ids : list[int], optional
            List of IDs corresponding to each point.
            If None, the points are treated as a ball track.

        Returns
        -------
        dict[int, np.ndarray] or np.ndarray
            Smoothed pitch coordinates per ID (dict) or for the ball (array).
        """
        if points.shape[0] == 0:
            return {} if player_ids is not None else np.empty((0, 2), dtype=np.float32)

        transformed = self._transform_raw(points)

        # Case 1: Ball (single track without IDs)
        if player_ids is None:
            z = transformed[0]
            f = self.filters.get(self._ball_id)
            if f is None:
                self.filters[self._ball_id] = self._init_kf(z)
                return transformed
            smoothed = self._kf_update(f, z)
            return smoothed.reshape(1, 2)

        # Case 2: Players (tracks with IDs)
        result: dict[int, np.ndarray] = {}
        for curr_pos, player_id in zip(transformed, player_ids):
            prev_pos = self.last_valid_positions.get(player_id)

            # Outlier rejection: skip if displacement exceeds threshold
            if prev_pos is not None:
                distance = float(np.linalg.norm(curr_pos - prev_pos))
                if distance > self.distance_threshold:
                    result[player_id] = prev_pos
                    continue

            # Kalman filter update
            z = curr_pos.astype(np.float32)
            f = self.filters.get(player_id)
            if f is None:
                self.filters[player_id] = self._init_kf(z)
                self.last_valid_positions[player_id] = z
                result[player_id] = z
            else:
                smoothed = self._kf_update(f, z)
                self.last_valid_positions[player_id] = smoothed
                result[player_id] = smoothed

        return result
