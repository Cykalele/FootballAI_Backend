import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # Ensures matplotlib runs in headless environments (e.g., FastAPI, servers)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration
from sports.annotators.soccer import draw_pitch
from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.interpolate import interp1d

# Number of interpolated points for paths
INTERPOLATED_POINTS = 50
# Minimum path length (in points) required for interpolation
MIN_LENGTH = 10


class HeatMapAnnotator:
    """
    Generates heatmaps from football tracking data.

    Supports team-based, player-specific, and ball-specific heatmaps,
    with options for time segmentation and KDE-based spatial density visualization.
    """

    def __init__(self, tracking_log, team_config, team_path, heatmap_path, mode="team", player_id=None):
        """
        Initializes the heatmap annotator and loads necessary configuration and data.

        Args:
            tracking_log (str): Path to JSON file containing tracking data (frames, player and ball positions).
            team_config (dict): Dictionary mapping team IDs to names and optional visual properties.
            team_path (str): Path to JSON file containing player-to-team assignments.
            heatmap_path (str): Directory to store generated heatmaps.
            mode (str): "team", "player", or "ball" – determines what the heatmap visualizes.
            player_id (int, optional): Player ID to focus on when mode="player".
        """
        self.mode = mode
        self.player_id = player_id if player_id else None
        self.pitch_config = SmallFieldConfiguration()
        self.width = self.pitch_config.width
        self.length = self.pitch_config.length
        self.team_config = team_config
        self.heatmap_path = heatmap_path

        # Load tracking and team assignment data
        self.tracking_data = self._load_json(tracking_log)
        self.team_assignments = self._load_json(team_path)

        # Ensure output directory exists
        os.makedirs(heatmap_path, exist_ok=True)

    def _load_json(self, path):
        """
        Loads JSON data from a file.

        Args:
            path (str): File path to load.

        Returns:
            dict: Parsed JSON object or empty dict if file not found.
        """
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {path}")
            return {}

    def _get_positions(self):
        """
        Extracts 2D positions from tracking data depending on the selected mode.

        For mode="player", only positions for the given player_id are returned.
        For mode="ball", only the ball's positions are returned.

        Returns:
            np.ndarray: Array of positions as (x, y) coordinates in centimeters.
        """
        pos = []

        for frame in self.tracking_data.values():
            players = frame.get("players", [])
            ball = frame.get("ball", [])

            if self.mode == "player" and self.player_id:
                for p in players:
                    if (
                        p["id"] == self.player_id
                        and "position_2d" in p
                        and isinstance(p["position_2d"], list)
                        and len(p["position_2d"]) == 2
                        and all(coord is not None for coord in p["position_2d"])
                    ):
                        pos.append(p["position_2d"])

            elif self.mode == "ball":
                if (
                    ball
                    and "position_2d" in ball[0]
                    and isinstance(ball[0]["position_2d"], list)
                    and len(ball[0]["position_2d"]) == 2
                    and all(
                        isinstance(coord, (float, int)) and coord is not None
                        for coord in ball[0]["position_2d"])
                ):
                    pos.append(ball[0]["position_2d"])

        return np.array(pos, dtype=np.float32)

    def generate_team_minute_heatmaps(self, frames_per_minute=1800):
        """
        Generates KDE-based heatmaps for each team, split into minute intervals.

        Args:
            frames_per_minute (int): Number of frames corresponding to one minute.
        """
        print("[ENTRY] generate_team_minute_heatmaps called")

        os.makedirs(self.heatmap_path, exist_ok=True)

        # Determine numeric frame indices
        frame_keys = sorted([int(k) for k in self.tracking_data.keys() if k.isdigit()])
        if not frame_keys:
            print("[WARN] No numeric frame keys found.")
            return

        max_frame = max(frame_keys)
        # Create (start, end) ranges for each minute
        minute_ranges = [(i, i + frames_per_minute) for i in range(0, max_frame + 1, frames_per_minute)]

        # Process each team
        for team_id, team_info in self.team_config.items():
            team_name = team_info.get("name", f"team_{team_id}")

            # Process each time segment
            for idx, (start, end) in enumerate(minute_ranges):
                positions = []
                for frame_idx in range(start, min(end, max_frame + 1)):
                    frame = self.tracking_data.get(str(frame_idx), {})
                    for p in frame.get("players", []):
                        pid = str(p.get("id"))
                        pos = p.get("position_2d")

                        player_info = self.team_assignments.get("players", {}).get(pid)
                        if not player_info:
                            continue
                        if str(player_info.get("team")) != str(team_id):
                            continue
                        if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(c, (int, float)) for c in pos)):
                            continue

                        positions.append(pos)

                if not positions:
                    continue

                positions = np.array(positions)
                x = positions[:, 0]
                y = self.width - positions[:, 1]  # Flip vertically for correct pitch orientation

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(draw_pitch(self.pitch_config), extent=[0, self.length, 0, self.width], zorder=0)
                sns.kdeplot(x=x, y=y, fill=True, cmap="hot", bw_adjust=0.7, levels=100, alpha=0.7, ax=ax)
                ax.set_xlim(0, self.length)
                ax.set_ylim(0, self.width)
                ax.axis("off")

                save_name = f"heatmap_team_{team_name}_min_{idx:02d}.png"
                save_path = os.path.join(self.heatmap_path, save_name)
                fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

    def generate_player_or_danger_heatmap(self):
        """
        Generates a single KDE heatmap for a specific player or the ball,
        depending on the current mode.
        """
        positions = self._get_positions()

        if len(positions) == 0:
            print("[WARN] No positions found.")
            return

        x = positions[:, 0]
        y = self.width - positions[:, 1]

        pitch = draw_pitch(self.pitch_config)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(pitch, extent=[0, self.length, 0, self.width], zorder=0)

        sns.kdeplot(x=x, y=y, cmap="hot", fill=True, bw_adjust=0.7, levels=100, alpha=0.7, thresh=0.15, ax=ax)
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        ax.axis('off')

        filename = f"heatmap_{self.mode}.png"
        if self.mode == "player":
            filename = f"heatmap_player_{str(self.player_id)}.png"

        save_path = os.path.join(self.heatmap_path, filename)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def generate_team_heatmap_single(self, team_id: int):
        """
        Generates a single KDE heatmap for all positions of a given team.

        Args:
            team_id (int): Team ID for which to generate the heatmap.
        """
        all_positions = []

        for frame in self.tracking_data.values():
            for p in frame.get("players", []):
                pid = p["id"]
                player_info = self.team_assignments.get("players", {}).get(str(pid))

                if player_info and player_info.get("team") == team_id and "position_2d" in p:
                    all_positions.append(p["position_2d"])

        if not all_positions:
            return  # No position data available

        positions = np.array(all_positions)
        x_all = positions[:, 0]
        y_all = self.width - positions[:, 1]

        team_name = self.team_config[str(team_id)]["name"]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(draw_pitch(self.pitch_config), extent=[0, self.length, 0, self.width], zorder=0)

        sns.kdeplot(
            x=x_all,
            y=y_all,
            fill=True,
            cmap="hot",
            bw_adjust=0.5,
            thresh=0.4,
            levels=100,
            alpha=0.4,
            ax=ax,
            zorder=2
        )

        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        ax.axis("off")

        save_path = os.path.join(self.heatmap_path, f"heatmap_team_{team_name}.png")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def interpolate_path(self, path, num_points=INTERPOLATED_POINTS):
        """
        Interpolates a given movement path to a fixed number of points using linear interpolation.

        Args:
            path (list or np.ndarray): Original sequence of (x, y) positions.
            num_points (int): Number of interpolated points to generate.

        Returns:
            np.ndarray or None: Interpolated path as (num_points, 2) or None if interpolation fails.
        """
        if len(path) < MIN_LENGTH:
            return None

        path = np.array(path)
        t_original = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_points)

        try:
            fx = interp1d(t_original, path[:, 0], kind='linear')
            fy = interp1d(t_original, path[:, 1], kind='linear')
            x_interp = fx(t_new)
            y_interp = fy(t_new)
            return np.column_stack((x_interp, y_interp))
        except Exception as e:
            print(f"[WARN] Interpolation failed: {e}")
            return None


def hex_to_cmap(hex_color, name="custom_cmap"):
    """
    Converts a hex color string into a matplotlib colormap
    ranging from white to the specified color.

    Args:
        hex_color (str): Hexadecimal color code (e.g., "#FF0000").
        name (str): Name for the generated colormap.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The resulting colormap.
    """
    rgb = mcolors.to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(name, ["white", rgb], N=256)
