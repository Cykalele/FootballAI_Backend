import numpy as np
from typing import Dict
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration

def filter(pitch_player_coords_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Filters out players who are located outside the pitch boundaries.

    Args:
        pitch_player_coords_dict (Dict[int, np.ndarray]): 
            A dictionary mapping player IDs to their 2D coordinates [x, y] in centimeters.

    Returns:
        Dict[int, np.ndarray]: 
            A filtered dictionary containing only players whose positions lie within the pitch dimensions.
    """
    # Load pitch configuration (length and width in cm)
    config = SmallFieldConfiguration()
    pitch_length = config.length
    pitch_width = config.width

    # Only include players whose x and y coordinates lie within the pitch bounds
    filtered = {
        pid: coord for pid, coord in pitch_player_coords_dict.items()
        if 0 <= coord[0] <= pitch_length and 0 <= coord[1] <= pitch_width
    }
    return filtered

def filter_ball_coordinates(ball_coords: np.ndarray) -> np.ndarray:
    """
    Filters out ball positions that lie outside the pitch boundaries.

    Args:
        ball_coords (np.ndarray): Nx2 array of ball positions in cm (pitch coordinates).

    Returns:
        np.ndarray: Filtered ball positions inside the pitch, or empty if none valid.
    """
    config = SmallFieldConfiguration()
    pitch_length = config.length
    pitch_width = config.width

    if ball_coords.ndim != 2 or ball_coords.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    valid = (
        (0 <= ball_coords[:, 0]) & (ball_coords[:, 0] <= pitch_length) &
        (0 <= ball_coords[:, 1]) & (ball_coords[:, 1] <= pitch_width)
    )
    return ball_coords[valid]

