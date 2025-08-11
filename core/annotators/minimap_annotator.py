from typing import Dict, Union, Tuple
import cv2
import numpy as np
import supervision as sv

from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration
from core.view_transformation.utils.pitch_utils import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)


class MinimapAnnotator:
    """
    Provides methods for creating tactical minimap visualizations, including 
    player locations, ball position, and Voronoi diagrams representing pitch control.
    """

    def __init__(self, team_config: Dict):
        """
        Initializes the MinimapAnnotator.

        Args:
            team_config (dict): Dictionary containing color and team mapping configuration.
        """
        self.pitch_config = SmallFieldConfiguration()
        self.team_config = team_config
        self.cached_pitch_background = None

    def annotate_simple_minimap(
        self,
        frame_idx: int,
        ball_xy: np.ndarray,
        player_coords_dict: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Generates a simplified pitch visualization containing only player and ball positions. 
        No team-specific coloring or Voronoi diagrams are included. 
        Primarily used for evaluating view transformation accuracy.

        Args:
            frame_idx (int): Index of the current frame.
            ball_xy (np.ndarray): Two-dimensional ball coordinates.
            player_coords_dict (dict): Mapping of player_id to {"position_2d": np.ndarray}.

        Returns:
            np.ndarray: Pitch image with annotated player and ball positions.
        """
        pitch = self._get_cached_pitch_background()

        # Draw ball position if available
        if ball_xy is not None:
            pitch = draw_points_on_pitch(
                self.pitch_config,
                ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=pitch
            )

        # Draw all players in a neutral color (e.g., red)
        all_positions = np.array(
            [v["position_2d"] for v in player_coords_dict.values()],
            dtype=np.float32
        )
        pitch = draw_points_on_pitch(
            self.pitch_config,
            all_positions,
            face_color=sv.Color.RED,
            edge_color=sv.Color.BLACK,
            radius=12,
            pitch=pitch
        )

        return pitch

    def annotate_pitch_diagramm(
        self,
        frame_idx: int,
        ball_xy: np.ndarray,
        player_coords_team_dict: Dict[int, Dict[str, Union[np.ndarray, int]]]
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Creates a pitch visualization with Voronoi pitch control for both teams 
        and overlays player and ball positions.

        Args:
            frame_idx (int): Index of the current frame.
            ball_xy (np.ndarray): Two-dimensional ball coordinates.
            player_coords_team_dict (dict): Mapping of player_id to a dictionary containing:
                - "position_2d" (np.ndarray): Player position in pitch coordinates.
                - "team" (int): Team identifier (0 for team 1, 1 for team 2).

        Returns:
            tuple:
                np.ndarray: Pitch image with annotated pitch control, players, and ball.
                dict: Frame-wise control mask mapping frame index to mask array.
        """
        pitch = self._get_cached_pitch_background()
        framewise_control_mask = {}

        # Separate positions by team
        team_1_xy = [v["position_2d"] for v in player_coords_team_dict.values() if v["team"] == 0]
        team_2_xy = [v["position_2d"] for v in player_coords_team_dict.values() if v["team"] == 1]

        team_1_xy = np.array(team_1_xy, dtype=np.float32)
        team_2_xy = np.array(team_2_xy, dtype=np.float32)

        # Compute Voronoi control areas
        pitch, control_mask = draw_pitch_voronoi_diagram(
            config=self.pitch_config,
            team_1_xy=team_1_xy,
            team_2_xy=team_2_xy,
            team_1_color=sv.Color.from_hex(self.team_config["1"]["color"]),
            team_2_color=sv.Color.from_hex(self.team_config["2"]["color"]),
            pitch=pitch,
            return_mask=True
        )

        total_pixels = control_mask.size
        team_1_control = np.sum(control_mask == 1) / total_pixels * 100
        team_2_control = 100 - team_1_control
        framewise_control_mask[frame_idx] = control_mask

        # Draw ball position if available
        if ball_xy is not None:
            pitch = draw_points_on_pitch(
                self.pitch_config,
                ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=pitch
            )

        # Draw player positions with team-specific colors
        pitch = draw_points_on_pitch(
            self.pitch_config,
            team_1_xy,
            face_color=sv.Color.from_hex(self.team_config["1"]["color"]),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )
        pitch = draw_points_on_pitch(
            self.pitch_config,
            team_2_xy,
            face_color=sv.Color.from_hex(self.team_config["2"]["color"]),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )

        return pitch, framewise_control_mask

    def overlay_minimap(
        self,
        main_frame: np.ndarray,
        minimap: np.ndarray,
        position: str = "top-right",
        alpha: float = 0.8,
        scale: float = 0.35
    ) -> np.ndarray:
        """
        Places the minimap as an overlay onto the main video frame.

        Args:
            main_frame (np.ndarray): The main video frame.
            minimap (np.ndarray): The minimap image to overlay.
            position (str): One of {"top-right", "top-left", "bottom-right", "bottom-left"}.
            alpha (float): Transparency of the overlay.
            scale (float): Scaling factor for the minimap.

        Returns:
            np.ndarray: Frame with the minimap overlay applied.
        """
        minimap = cv2.resize(minimap, (0, 0), fx=scale, fy=scale)
        h_mini, w_mini = minimap.shape[:2]
        h_main, w_main = main_frame.shape[:2]

        if position == "top-right":
            x, y = w_main - w_mini - 10, 10
        elif position == "top-left":
            x, y = 10, 10
        elif position == "bottom-right":
            x, y = w_main - w_mini - 10, h_main - h_mini - 10
        elif position == "bottom-left":
            x, y = 10, h_main - h_mini - 10
        else:
            raise ValueError(f"Invalid position: {position}")

        roi = main_frame[y:y + h_mini, x:x + w_mini]
        blended = cv2.addWeighted(roi, 1 - alpha, minimap, alpha, 0)
        main_frame[y:y + h_mini, x:x + w_mini] = blended
        return main_frame

    def _get_cached_pitch_background(self) -> np.ndarray:
        """
        Returns a copy of the cached pitch background. If not cached yet, 
        it is drawn once and stored for reuse.

        Returns:
            np.ndarray: Cached pitch background image.
        """
        if self.cached_pitch_background is None:
            self.cached_pitch_background = draw_pitch(self.pitch_config)
        return self.cached_pitch_background.copy()
