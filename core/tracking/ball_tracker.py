import time
import cv2
import numpy as np
from collections import deque
import supervision as sv
from typing import Deque, Optional


class BallTracker:
    """
    Tracks the football using direct YOLO detections and applies optical flow
    as a fallback when a reliable detection is not available. The tracker
    performs outlier rejection, plausibility checks on motion, and resets
    its internal state after prolonged loss of the ball.
    """

    def __init__(self, buffer_size: int = 10, max_distance_ratio: float = 0.15, min_confidence: float = 0.4) -> None:
        """
        Initializes the tracker state and threshold parameters.

        Parameters
        ----------
        buffer_size
            The number of recent positions stored for smoothing and velocity calculation.
        max_distance_ratio
            The maximum allowed displacement per frame expressed as a fraction of the image diagonal.
        min_confidence
            The minimum confidence required to accept a ball detection as valid.
        """
        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)   # Recent positions of the ball center
        self.prev_frame: Optional[np.ndarray] = None                 # Previous full BGR frame
        self.prev_gray: Optional[np.ndarray] = None                  # Previous grayscale frame
        self.prev_ball_position: Optional[np.ndarray] = None         # Last valid ball center (x, y)

        self.max_distance_ratio = max_distance_ratio                 # Displacement threshold relative to image diagonal
        self.min_confidence = min_confidence                         # Minimum detection confidence

        self.frame_counter = 0                                       # Total processed frames
        self.frame_counter_since_last_valid = 0                      # Frames since last valid position

    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        Processes the current frame and returns the tracked ball for this frame.
        The method prioritizes direct detections and falls back to optical flow
        if no reliable detection is available.

        Parameters
        ----------
        frame
            The current video frame as a BGR image.
        detections
            YOLO detections for the ball in the current frame.

        Returns
        -------
        sv.Detections
            A Supervision Detections object containing the selected ball box.
            An empty Detections object is returned if the ball cannot be established.
        """
        self.frame_counter += 1
        height, width = frame.shape[:2]
        max_distance = self.max_distance_ratio * np.sqrt(width ** 2 + height ** 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        # Try to select the best detection by maximum confidence
        if detections and detections.confidence is not None and len(detections.confidence) > 0:
            best_idx = np.argmax(detections.confidence)
            best_conf = detections.confidence[best_idx]
            best_pos = xy[best_idx]

            # Outlier rejection based on maximum plausible displacement
            if best_conf >= self.min_confidence:
                if (
                    self.prev_ball_position is None
                    or np.linalg.norm(best_pos - self.prev_ball_position) <= max_distance
                ):
                    # Accept new position and refresh state
                    self.prev_ball_position = best_pos
                    self.prev_gray = gray.copy()
                    self.prev_frame = frame.copy()
                    self.buffer.append(best_pos)
                    self.frame_counter_since_last_valid = 0

                    return sv.Detections(
                        xyxy=np.array([detections.xyxy[best_idx]]),
                        confidence=np.array([best_conf]),
                    )

        # Optical flow fallback when detection is missing or unreliable
        if self.prev_ball_position is not None and self.prev_gray is not None:
            p0 = np.array([self.prev_ball_position], dtype=np.float32).reshape(-1, 1, 2)
            try:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)

                if status[0]:
                    predicted = p1[0][0]

                    # Plausibility check on the predicted motion
                    if np.linalg.norm(predicted - self.prev_ball_position) <= max_distance:
                        self.prev_ball_position = predicted
                        self.prev_gray = gray.copy()
                        self.prev_frame = frame.copy()
                        self.buffer.append(predicted)
                        self.frame_counter_since_last_valid += 1

                        return sv.Detections(
                            xyxy=np.array(
                                [[predicted[0] - 3, predicted[1] - 3, predicted[0] + 3, predicted[1] + 3]]
                            ),
                            confidence=np.array([0.0]),
                        )
            except Exception as e:
                print(f"[BallTracker] Optical flow error: {e}")

        self.frame_counter_since_last_valid += 1

        # Recovery attempt after a moderate duration without a valid position
        if 15 <= self.frame_counter_since_last_valid < 40:
            # Reconsider weaker candidates with a lower confidence threshold
            fallback_candidates = [
                (conf, pos, xyxy_box)
                for conf, pos, xyxy_box in zip(detections.confidence, xy, detections.xyxy)
                if conf >= 0.2 and np.linalg.norm(pos - self.prev_ball_position) <= max_distance
            ]
            if fallback_candidates:
                fallback_candidates.sort(key=lambda x: x[0], reverse=True)
                best_conf, best_pos, best_box = fallback_candidates[0]

                self.prev_ball_position = best_pos
                self.prev_gray = gray.copy()
                self.prev_frame = frame.copy()
                self.buffer.append(best_pos)
                self.frame_counter_since_last_valid = 0

                return sv.Detections(xyxy=np.array([best_box]), confidence=np.array([best_conf]))

        # Full reset after prolonged loss
        if self.frame_counter_since_last_valid >= 40:
            print("[BallTracker] Resetting tracker after prolonged loss.")
            self.prev_ball_position = None
            self.prev_gray = None
            self.prev_frame = None
            self.buffer.clear()
            self.frame_counter_since_last_valid = 0

        # No valid ball could be established in this frame
        return sv.Detections.empty()
