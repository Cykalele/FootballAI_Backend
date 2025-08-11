from collections import deque
import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    Annotates the football (ball) in video frames with a visual trail.
    The trail consists of filled or outlined circles whose size and color
    vary with the age of the recorded ball position, producing a fading effect.
    """

    def __init__(self, radius: int = 5, buffer_size: int = 5, thickness: int = -1, transparency: float = 0.7):
        """
        Initializes the ball annotator and allocates the position buffer.

        Parameters
        ----------
        radius : int, optional
            Maximum radius in pixels for the most recent ball position.
        buffer_size : int, optional
            Number of past ball positions to store and render as a trail.
        thickness : int, optional
            Thickness of the circle in pixels (-1 for filled circles).
        transparency : float, optional
            Reserved for future alpha blending of the ball overlay. Not used in the current implementation.
        """
        # Color palette mapped to the buffer length; older positions are assigned cooler colors.
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)

        # Stores the most recent ball positions as arrays of center points.
        self.buffer = deque(maxlen=buffer_size)

        self.radius = radius
        self.thickness = thickness
        self.transparency = transparency

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Computes the radius for a trail circle given its relative age.

        Parameters
        ----------
        i : int
            Index of the current position in the trail, with 0 representing the oldest.
        max_i : int
            Total number of positions in the trail.

        Returns
        -------
        int
            Circle radius in pixels for the current index.
        """
        if max_i == 1:
            return self.radius
        # Linear interpolation from 1 pixel to the maximum radius.
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Draws the ball and its recent positions as a fading trail onto the frame.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame in BGR format.
        detections : sv.Detections
            Ball detections for the current frame; only the bottom center anchor is used for positioning.

        Returns
        -------
        np.ndarray
            The annotated frame with the ball trail drawn on top.
        """
        # Extract bottom-center anchor points of each detection as integers.
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)

        # Append the current positions to the position buffer.
        self.buffer.append(xy)

        # Iterate over buffered positions from oldest to newest.
        buffer_copy = list(self.buffer)
        for i, xy_points in enumerate(buffer_copy):
            color = self.color_palette.by_idx(i)
            radius = self.interpolate_radius(i, len(buffer_copy))
            for center in xy_points:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame
