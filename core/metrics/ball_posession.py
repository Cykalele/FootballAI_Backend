from typing import Dict, Tuple
import numpy as np
from collections import defaultdict
import json

class BallPossessionAnalyzer:
    """
    Analyzes ball possession by assigning the closest player to the ball.
    Tracks framewise possession percentages, player/team in possession,
    and calculates pass counts.
    """

    def __init__(self, assignment_path):
        """
        Initializes the ball possession analyzer.

        Args:
            assignment_path (str): Path to JSON file with team/player assignments.
        """
        self.possession_count = defaultdict(int)
        self.total_frames = 0
        self.framewise_percentages = {}
        self.framewise_possession_log = {}  # frame_idx -> {player_id, team, position}
        self.framewise_pass_counts = {}  # frame_idx -> (team1_passes, team2_passes)

        self.candidate_possessor = None
        self.candidate_count = 0

    def update(self, frame_idx: int, ball_coords, pitch_player_coords_dict: dict[int, np.ndarray],
               player_ids: list[int], team_assignments: dict[int, int]):
        """
        Updates ball possession information for a given frame.

        The closest player to the ball is assigned as possessor.
        Possession changes only if a new player is at least 5% closer.

        Args:
            frame_idx (int): Current frame index.
            ball_coords (List[np.ndarray]): Ball position as 2D coordinates.
            pitch_player_coords_dict (dict): Mapping of player_id to 2D positions.
            player_ids (list[int]): List of visible player IDs in the frame.
            team_assignments (dict[int, int]): Mapping from player_id to team_id.
        """
        # Handle missing ball coordinates or invalid data
        if (
            ball_coords is None or
            len(ball_coords) == 0 or
            not pitch_player_coords_dict or
            np.any(np.isnan(ball_coords))
        ):
            # Fallback to last known possession percentages or zero if none exist
            if self.framewise_percentages:
                last_frame = max(self.framewise_percentages.keys())
                self.framewise_percentages[int(frame_idx)] = self.framewise_percentages[last_frame]
                self.framewise_possession_log[str(frame_idx)] = self.framewise_possession_log.get(
                    str(frame_idx - 1),
                    {"player_id": None, "team": None, "position": None}
                )
            else:
                self.framewise_percentages[int(frame_idx)] = (0.0, 0.0)
            return

        # Keep only valid players that exist in the coordinate dictionary
        valid_entries = [
            (int(pid), pitch_player_coords_dict[int(pid)])
            for pid in player_ids
            if int(pid) in pitch_player_coords_dict
        ]
        if not valid_entries:
            return

        player_ids_filtered, player_coords_array = zip(*valid_entries)
        player_coords_array = np.array(player_coords_array, dtype=np.float32)

        # Calculate distances from ball to each player
        distances = np.linalg.norm(player_coords_array - ball_coords[0], axis=1)
        closest_idx = np.argmin(distances)
        closest_player_id = str(player_ids_filtered[closest_idx])
        closest_distance = distances[closest_idx]

        # Retrieve last possessor information
        last_entry = self.framewise_possession_log.get(str(frame_idx - 1), {
            "player_id": None, "team": None, "position": None
        })
        last_owner = last_entry["player_id"]
        last_team = last_entry["team"]
        last_distance = None

        if last_owner is not None and int(last_owner) in player_ids_filtered:
            last_owner_idx = player_ids_filtered.index(int(last_owner))
            last_distance = np.linalg.norm(player_coords_array[last_owner_idx] - ball_coords[0])

        # Decide whether to switch possession
        if last_distance is None or closest_distance < 1.05 * last_distance:
            if int(closest_player_id) in team_assignments:
                team = int(team_assignments[int(closest_player_id)])
                self.possession_count[team] += 1
                self.framewise_possession_log[str(frame_idx)] = {
                    "player_id": closest_player_id,
                    "team": team,
                    "position": [float(x) for x in ball_coords[0]]
                }
            else:
                self.framewise_possession_log[str(frame_idx)] = {
                    "player_id": None,
                    "team": None,
                    "position": [float(x) for x in ball_coords[0]]
                }
        else:
            # Keep previous possessor
            self.framewise_possession_log[str(frame_idx)] = {
                "player_id": last_owner,
                "team": last_team,
                "position": [float(x) for x in ball_coords[0]]
            }

        # Update possession percentages
        total = sum(self.possession_count.values())
        team1_pct = self.possession_count[1] / total * 100 if total > 0 else 0.0
        team2_pct = self.possession_count[2] / total * 100 if total > 0 else 0.0
        self.framewise_percentages[int(frame_idx)] = (float(team1_pct), float(team2_pct))
        self.total_frames += 1

    def get_possessor_by_frame(self, frame_idx: int):
        """
        Returns the player ID possessing the ball at a given frame.

        Args:
            frame_idx (int): Frame index.

        Returns:
            str or None: Player ID or None if unavailable.
        """
        return self.framewise_possession_log.get(str(frame_idx), {}).get('player_id', None)

    def get_ball_posession(self):
        """
        Returns framewise possession percentages for both teams.

        Returns:
            dict[int, Tuple[float, float]]: Frame-indexed percentages.
        """
        return self.framewise_percentages

    def get_framewise_teamwise_pass_counts_by_frame(self) -> Dict[int, Tuple[int, int]]:
        """
        Calculates the cumulative number of passes for each team at every frame.

        Returns:
            dict[int, Tuple[int, int]]: Mapping frame index -> (team1_passes, team2_passes)
        """
        passes_team1 = 0
        passes_team2 = 0
        frame_pass_count = {}
        last_pass_frame = -100
        min_frames_between_passes = 5
        min_velocity_threshold = 10
        min_possession_duration = 5
        last_owner = None
        possession_counter = 0

        sorted_frames = sorted(self.framewise_possession_log.keys(), key=int)

        for i in range(1, len(sorted_frames)):
            frame = int(sorted_frames[i])
            curr_entry = self.framewise_possession_log[sorted_frames[i]]
            prev_entry = self.framewise_possession_log[sorted_frames[i - 1]]

            curr_pid, curr_team, curr_pos = curr_entry["player_id"], curr_entry["team"], curr_entry["position"]
            prev_pid, prev_team, prev_pos = prev_entry["player_id"], prev_entry["team"], prev_entry["position"]

            if curr_pid == prev_pid:
                possession_counter += 1
            else:
                if (
                    prev_pid and curr_pid and
                    possession_counter >= min_possession_duration and
                    prev_team in [1, 2] and curr_team in [1, 2] and
                    curr_team == prev_team and
                    curr_pos and prev_pos
                ):
                    velocity = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                    if velocity >= min_velocity_threshold and (frame - last_pass_frame) >= min_frames_between_passes:
                        if prev_team == 1:
                            passes_team1 += 1
                        elif prev_team == 2:
                            passes_team2 += 1
                        last_pass_frame = frame
                possession_counter = 1

            frame_pass_count[frame] = (passes_team1, passes_team2)
            self.framewise_pass_counts = frame_pass_count

        return frame_pass_count

    def get_total_passes(self):
        """
        Returns the total passes for each team.

        Returns:
            dict[int, int]: {team_id: passes}
        """
        last = self.get_framewise_teamwise_pass_counts_by_frame()
        if not last:
            return {1: 0, 2: 0}
        last_frame = max(last.keys())
        team1, team2 = last[last_frame]
        return {1: team1, 2: team2}

    def get_percentages(self):
        """
        Returns overall possession percentages for each team.

        Returns:
            dict[int, float]: {team_id: percentage}
        """
        total = sum(self.possession_count.values())
        return {
            1: round(self.possession_count[1] / total * 100, 2) if total else 0.0,
            2: round(self.possession_count[2] / total * 100, 2) if total else 0.0
        }

    def save(self, ball_possession_path, ball_events_path):
        """
        Saves possession summary to ball_possession_path
        and all pass-related events to ball_events_path.

        Args:
            ball_possession_path (str): Path to save possession summary JSON.
            ball_events_path (str): Path to save ball events JSON.
        """
        # Convert possession log for JSON serialization
        summary = self.get_percentages()
        converted_log = {
            frame: {
                "player_id": entry["player_id"],
                "team": int(entry["team"]) if entry["team"] is not None else None,
                "position": [float(x) for x in entry["position"]] if entry["position"] else None
            }
            for frame, entry in self.framewise_possession_log.items()
        }

        # Save ball_possession.json
        possession_data = {
            "frames": converted_log,
            "summary": {
                "team_1_possession_percent": float(summary[1]),
                "team_2_possession_percent": float(summary[2])
            }
        }
        with open(ball_possession_path, "w") as f:
            json.dump(possession_data, f, indent=4)

        # Load existing ball_events.json if available
        try:
            with open(ball_events_path, "r") as f:
                ball_events_data = json.load(f)
        except FileNotFoundError:
            ball_events_data = {}

        # Update passes information
        ball_events_data["passes"] = {
            "total": {
                "1": self.get_total_passes()[1],
                "2": self.get_total_passes()[2]
            },
            "framewise": {
                str(k): [int(v[0]), int(v[1])]
                for k, v in self.framewise_pass_counts.items()
            }
        }

        # Save updated ball_events.json
        with open(ball_events_path, "w") as f:
            json.dump(ball_events_data, f, indent=4)
