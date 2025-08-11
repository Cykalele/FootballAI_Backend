import numpy as np
import json
from collections import defaultdict

class PlayerMetricsAnalyzer:
    """
    Tracks and analyzes player movement metrics such as distance covered and speed.

    Functionality:
    - Records per-frame player positions in pitch coordinates.
    - Computes framewise distances, instantaneous speeds, and total distances.
    - Speeds are stored in both meters per second (m/s) and kilometers per hour (km/h).
    """

    def __init__(self):
        # Data structure: player_id → metrics dictionary
        self.player_data = defaultdict(lambda: {
            "positions": {},        # frame_idx → [x, y] position in meters
            "distances": {},        # frame_idx → float distance covered in meters (from previous frame)
            "speeds_kmh": {},       # frame_idx → speed in km/h
            "speeds_m/s": {},       # frame_idx → speed in m/s
            "total_distance": 0.0   # cumulative distance in meters
        })
        self.fps = 30  # Frames per second (must match the video frame rate)

    def update(self, frame_idx: int, player_positions_2d):
        """
        Updates movement metrics for all players in the current frame.

        Args:
            frame_idx (int): Current frame index in the video.
            player_positions_2d (list): List of tuples (player_id, np.array([x, y])),
                                         where positions are in centimeters.
        """
        for player_id, pos in player_positions_2d:
            pid = str(player_id)        # Store player IDs as strings for consistency
            pos = pos / 100.0           # Convert centimeters to meters

            # Store the player's position for this frame
            self.player_data[pid]["positions"][frame_idx] = pos.tolist()

            # If we have the previous frame's position, compute movement metrics
            if (frame_idx - 1) in self.player_data[pid]["positions"]:
                prev_pos = np.array(self.player_data[pid]["positions"][frame_idx - 1])

                # Displacement vector (meters) between current and previous position
                delta = pos - prev_pos

                # Euclidean distance covered in this frame (meters)
                dist = np.linalg.norm(delta)

                # Instantaneous speed in meters per second (distance × FPS)
                speed_mps = dist * self.fps

                # Store computed metrics
                self.player_data[pid]["distances"][frame_idx] = dist
                self.player_data[pid]["speeds_m/s"][frame_idx] = speed_mps
                self.player_data[pid]["speeds_kmh"][frame_idx] = speed_mps * 3.6

                # Update cumulative distance covered
                self.player_data[pid]["total_distance"] += dist

    def save(self, path):
        """
        Saves the collected metrics for all players to a JSON file.

        Args:
            path (str): Output file path where metrics will be stored.
        """
        output = {
            "players": {}
        }

        for pid, data in self.player_data.items():
            output["players"][pid] = {
                "total_distance_m": round(data["total_distance"], 2),
                "positions": data["positions"],
                "distances": {str(f): round(d, 4) for f, d in data["distances"].items()},
                "speeds_kmh": {str(f): round(s, 2) for f, s in data["speeds_kmh"].items()},
                "speeds_m/s": {str(f): round(s, 2) for f, s in data["speeds_m/s"].items()}
            }

        # Write the JSON file with indentation for readability
        with open(path, "w") as f:
            json.dump(output, f, indent=4)
