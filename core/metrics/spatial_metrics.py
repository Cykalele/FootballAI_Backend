import numpy as np
import json
from collections import defaultdict
from typing import Dict, Tuple, List
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration

class SpatialAnalyzer:
    """
    Computes spatial control metrics based on a Voronoi-style control mask.
    """

    def __init__(self, assignment_path: str):
        self.pitch_config = SmallFieldConfiguration()
        self.pitch_width = self.pitch_config.width  # in cm
        self.pitch_length = self.pitch_config.length  # in cm

        self.metrics = defaultdict(dict)

    def update(self, frame_idx: int, framewise_control_mask: Dict[int, np.ndarray]):
        """
        Updates spatial metrics for a given frame based on control mask.

        Args:
            frame_idx (int): Frame index.
            framewise_control_mask (dict): Mapping from frame index to control mask array.
        """
        self.set_control_percentages(frame_idx, framewise_control_mask)
        self.calculate_thirds_control(frame_idx, framewise_control_mask)

    def set_control_percentages(self, frame_idx: int, framewise_control_mask: Dict[int, np.ndarray]) -> None:
        """
        Calculates overall control percentages for each team across the entire pitch.

        Args:
            frame_idx (int): Frame index.
            framewise_control_mask (dict): Frame index → control mask.
        """
        control_mask = framewise_control_mask[frame_idx]
        total_pixels = control_mask.size

        team_1_percentage = np.sum(control_mask == 1) / total_pixels * 100
        team_2_percentage = 100 - team_1_percentage

        self.metrics[str(frame_idx)]["space_control"] = (team_1_percentage, team_2_percentage)

    def get_control_percentages(self) -> Dict[int, Tuple[float, float]]:
        """
        Returns stored framewise control percentages for each team.

        Returns:
            dict: frame index → (team_1_percentage, team_2_percentage)
        """
        control = {}
        for frame_idx_str, frame_data in self.metrics.items():
            if "space_control" in frame_data:
                frame_idx = int(frame_idx_str)
                control[frame_idx] = tuple(frame_data["space_control"])
        return control

    def calculate_thirds_control(self, frame_idx: int, framewise_control_mask: Dict[int, np.ndarray], scale: float = 0.1, padding: int = 50) -> None:
        """
        Computes control percentages per pitch third (defensive, middle, attacking).

        Args:
            frame_idx (int): Frame index.
            framewise_control_mask (dict): Frame index → control mask.
            scale (float): Scaling factor to map pitch length to pixel space.
            padding (int): Horizontal offset (left side).
        """
        control_mask = framewise_control_mask[frame_idx]

        thirds_px = int(self.pitch_length * scale) // 3
        thirds_control = {
            "defensive": {1: 0, 2: 0},
            "middle": {1: 0, 2: 0},
            "attacking": {1: 0, 2: 0}
        }

        zones = {
            "defensive": slice(padding, padding + thirds_px),
            "middle": slice(padding + thirds_px, padding + 2 * thirds_px),
            "attacking": slice(padding + 2 * thirds_px, padding + 3 * thirds_px)
        }

        for zone_name, x_slice in zones.items():
            zone_mask = control_mask[x_slice]
            team1_pixels = np.sum(zone_mask == 1)
            team2_pixels = np.sum(zone_mask == 2)
            total_pixels = team1_pixels + team2_pixels

            if total_pixels > 0:
                thirds_control[zone_name][1] = round(team1_pixels / total_pixels * 100, 2)
                thirds_control[zone_name][2] = round(team2_pixels / total_pixels * 100, 2)

        self.metrics[str(frame_idx)]["thirds_control_percent"] = thirds_control

    def save(self, path: str) -> None:
        """
        Saves all spatial metrics to a JSON file.

        Args:
            path (str): Output path for metrics JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)
