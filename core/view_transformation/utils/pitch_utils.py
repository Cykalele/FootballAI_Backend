# SPDX-License-Identifier: MIT
# Adapted from: Roboflow "Sports" repository (MIT License)
# Source: https://github.com/roboflow/sports/tree/main
# This file is an adaptation for the FootballAI project. The original
# implementation from Roboflow Sports is licensed under the MIT License.
# All original copyright and permission notices must be retained in
# accordance with the license. A copy of the MIT License should be included
# in the FootballAI repository, and the README should credit the upstream
# source.  
#
# Changes in this adaptation:
# - Integrated into FootballAI's `SmallFieldConfiguration` pitch model.
# - Added optional mask output in `draw_pitch_voronoi_diagram` for
#   space-control metric calculations.
# - Minor code adjustments for compatibility with FootballAI pipeline.
#
# Original structure, naming, and visual drawing logic are preserved.

from typing import Optional, List, Tuple, Union
import cv2
import supervision as sv
import numpy as np

from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration

# Default global constants for scaling and layout padding
scale = 0.1
padding = 50


def draw_pitch(
    config: SmallFieldConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = padding,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = scale
) -> np.ndarray:
    """
    Draws a scaled football pitch with lines, penalty spots, and background.

    Parameters
    ----------
    config : SmallFieldConfiguration
        Field dimensions and vertex layout.
    background_color : sv.Color
        RGB color of the pitch surface.
    line_color : sv.Color
        RGB color of pitch lines and markings.
    padding : int
        Border margin in pixels around the pitch.
    line_thickness : int
        Thickness of all lines in pixels.
    point_radius : int
        Radius of penalty spots in pixels.
    scale : float
        Scaling factor from centimeters to pixels.

    Returns
    -------
    np.ndarray
        Rendered pitch image.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw all defined edges
    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # Draw the center circle
    centre_circle_center = (scaled_length // 2 + padding,
                            scaled_width // 2 + padding)
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    # Draw penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image


def draw_points_on_pitch(
    config: SmallFieldConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = padding,
    scale: float = scale,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points (e.g., player positions) on a pitch image.

    Parameters
    ----------
    config : SmallFieldConfiguration
        Field dimensions and layout.
    xy : np.ndarray
        Array of (x, y) positions in centimeters.
    face_color : sv.Color
        Fill color for the points.
    edge_color : sv.Color
        Outline color for the points.
    radius : int
        Circle radius in pixels.
    thickness : int
        Outline thickness in pixels.
    pitch : np.ndarray, optional
        Existing pitch image to overlay points on; if None, creates a new one.

    Returns
    -------
    np.ndarray
        Pitch image with points drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        scaled_point = (int(point[0] * scale) + padding,
                        int(point[1] * scale) + padding)
        cv2.circle(pitch, scaled_point, radius, face_color.as_bgr(), thickness=-1)
        cv2.circle(pitch, scaled_point, radius, edge_color.as_bgr(), thickness=thickness)

    return pitch


def draw_paths_on_pitch(
    config: SmallFieldConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = padding,
    scale: float = scale,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws polyline paths (e.g., player trajectories) on the pitch.

    Parameters
    ----------
    paths : list of np.ndarray
        Each path is an ordered list of (x, y) positions in centimeters.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for path in paths:
        scaled_path = [(int(point[0] * scale) + padding, int(point[1] * scale) + padding)
                       for point in path if point.size > 0]
        if len(scaled_path) < 2:
            continue
        for i in range(len(scaled_path) - 1):
            cv2.line(pitch, scaled_path[i], scaled_path[i + 1], color.as_bgr(), thickness)

    return pitch


def draw_pitch_voronoi_diagram(
    config: SmallFieldConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = padding,
    scale: float = scale,
    pitch: Optional[np.ndarray] = None,
    return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generates a Voronoi-based space-control diagram overlay for two teams.

    The pitch is divided such that each pixel is assigned to the nearest
    player of either team, producing a control mask.

    Parameters
    ----------
    return_mask : bool
        If True, also returns the raw control mask (1 = team 1 control,
        2 = team 2 control) in addition to the overlay.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)
    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coords, x_coords = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
    y_coords -= padding
    x_coords -= padding

    def calc_dists(xy, x_coords, y_coords):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coords) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coords) ** 2)

    d_team_1 = calc_dists(team_1_xy, x_coords, y_coords) if len(team_1_xy) > 0 else np.full((1, *x_coords.shape), np.inf)
    d_team_2 = calc_dists(team_2_xy, x_coords, y_coords) if len(team_2_xy) > 0 else np.full((1, *x_coords.shape), np.inf)

    min_team1 = np.min(d_team_1, axis=0)
    min_team2 = np.min(d_team_2, axis=0)
    control_mask = np.where(min_team1 < min_team2, 1, 2)

    voronoi[control_mask == 1] = team_1_color_bgr
    voronoi[control_mask == 2] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    if return_mask:
        return overlay, control_mask
    return overlay
