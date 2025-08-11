from typing import Dict, Tuple
import numpy as np
import json
import os
from collections import deque
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration


class GoalDetector:
    def __init__(self, assignment_path, verbose=False):
        """
        Initializes the GoalDetector with required configuration and internal buffers.

        Args:
            assignment_path (str): Path to the JSON file containing team assignments and directions.
            verbose (bool): Whether to print debug information.
        """
        self.verbose = verbose
        self.pitch_config = SmallFieldConfiguration()

        self.goals = {}
        self.shots = []
        self.framewise_shots = {}
        self.goal_counter = {"1": 0, "2": 0}

        self.attack_directions = self._load_attack_directions(assignment_path)

        self.recent_ball_positions = deque(maxlen=20)
        self.last_possessor_by_frame = {}

        self.required_goal_frames = 15
        self.left_goal_frames = deque(maxlen=self.required_goal_frames)
        self.right_goal_frames = deque(maxlen=self.required_goal_frames)
        self.goal_cooldown_until_frame = -1
        self.goal_cooldown_frames = 100

        self.last_goal_side = None

        self.goal_depth_tolerance = 2.0  # meters

    def _load_attack_directions(self, assignment_path: str) -> dict[str, str]:
        """
        Loads only the attack direction information from the team assignment JSON.

        Args:
            assignment_path (str): Path to the JSON file.

        Returns:
            dict: Mapping of team ID (as string) to attack direction ('left_to_right' or 'right_to_left').
        """
        try:
            with open(assignment_path, "r") as f:
                data = json.load(f)
                return data.get("directions", {})
        except FileNotFoundError:
            print(f"⚠️ {assignment_path} not found – no direction info available.")
            return {}

    def update(self, frame_idx: int, ball_pos: np.ndarray, player_positions: np.ndarray, ball_possession: str, team_assignments: dict[int,int]):
        """
        Updates the detector for the current frame with ball and player information. Also handles shot and goal detection.

        Args:
            frame_idx (int): Index of the current video frame.
            ball_pos (np.ndarray): Ball position in (x, y) format.
            player_positions (np.ndarray): Positions of all players (unused here but could be extended).
            ball_possession (str): ID of the player currently possessing the ball.
        """
        if ball_pos is None or len(ball_pos) < 2:
            return  # Skip frame if ball position is invalid

        # Append current ball position and store possession info
        self.recent_ball_positions.append(ball_pos)
        self.last_possessor_by_frame[frame_idx] = ball_possession

        # Identify shooter by searching backwards in time
        shooter = None
        for back in range(0, 30):
            shooter = self.last_possessor_by_frame.get(frame_idx - back)
            if shooter is not None:
                break

        # Resolve shooter's team
        if shooter is None or int(shooter) not in team_assignments:
            return  # Cannot proceed without shooter or team info

        shooter_team = int(team_assignments[int(shooter)])

        # Get direction of play for shooter's team
        team_direction = self.attack_directions.get(str(shooter_team))
        if team_direction is None:
            return

        # Detect potential shot based on recent movement
        self._detect_shot(frame_idx, shooter, team_assignments)

        # Prevent detecting another goal within cooldown interval
        if frame_idx < self.goal_cooldown_until_frame:
            return

        # Check if ball has entered the goal area
        ball_x, ball_y = ball_pos
        side = "left" if team_direction == "right_to_left" else "right"
        in_goal = self._is_ball_in_goal_area(ball_x, ball_y, side)

        # Append result to appropriate goal frame buffer
        buffer = self.left_goal_frames if side == "left" else self.right_goal_frames
        if in_goal:
            buffer.append(True)
        else:
            buffer.clear()

        # If enough consecutive detections and not already detected on this side
        if buffer.count(True) >= self.required_goal_frames and self.last_goal_side != side:
            self.last_goal_side = side
            scoring_team = self._get_team_for_goal_side(side)
            self._register_goal(frame_idx, scoring_team, team_assignments)
            buffer.clear()

        # Reset goal flag once the ball leaves the goal area
        if self.last_goal_side and not self._is_ball_behind_goal_line(ball_x, self.last_goal_side):
            self.last_goal_side = None

    def _detect_shot(self, frame_idx: int, shooter: str, team_assignments: dict[int,int]):
        """
        Detects and logs a shot if the ball is moving quickly and in the direction of the goal.

        Args:
            frame_idx (int): Current frame index.
            shooter (str): Player ID of the potential shooter.
        """
        if len(self.recent_ball_positions) < 2:
            return  # Not enough data to compute velocity

        # Compute velocity vector
        p1, p0 = self.recent_ball_positions[-1], self.recent_ball_positions[-2]
        velocity = np.linalg.norm(np.array(p1) - np.array(p0))

        # Identify shooter's team and direction
        if shooter is None or int(shooter) not in team_assignments:
            return  # Cannot proceed without shooter or team info

        shooter_team = int(team_assignments[int(shooter)])
        team_direction = self.attack_directions.get(str(shooter_team))

        # Get last recorded shot counts
        last_frame = max(self.framewise_shots.keys(), default=-1)
        last_counts = self.framewise_shots.get(last_frame, (0, 0))
        team1_count, team2_count = last_counts

        # If velocity suggests a shot
        if 70 <= velocity < 500 and shooter:
            shot_direction = self._infer_shot_direction()
            if shot_direction == team_direction:
                is_shot_on_goal = self._is_shot_towards_goal_box(shot_direction)
                if is_shot_on_goal:
                    # Check for duplicate shots in last 15 frames
                    is_duplicate = any(
                        shot["player"] == shooter and abs(shot["frame"] - frame_idx) <= 15
                        for shot in self.shots
                    )
                    if not is_duplicate:
                        # Increase shot count and save shot metadata
                        if shooter_team == 1:
                            team1_count += 1
                        elif shooter_team == 2:
                            team2_count += 1

                        shot = {
                            "frame": frame_idx,
                            "player": str(shooter),
                            "team": shooter_team,
                            "direction": team_direction,
                            "velocity": float(velocity),
                            "on_goal": True
                        }
                        self.shots.append(shot)

                        if self.verbose:
                            print(f"Velocity: {velocity:.2f} cm/frame")
                            print(f"Shot detected: {shot}")

        # Save shot count regardless of detection
        self.framewise_shots[frame_idx] = (team1_count, team2_count)

    def _is_shot_towards_goal_box(self, direction: str) -> bool:
        """
        Checks whether the ball is moving toward the goal box based on direction.

        Args:
            direction (str): The team's attack direction ('left_to_right' or 'right_to_left').

        Returns:
            bool: True if ball trajectory is directed into the goal box.
        """
        if len(self.recent_ball_positions) < 2:
            return False

        # Calculate shot vector
        p1 = self.recent_ball_positions[-1]
        p0 = self.recent_ball_positions[-2]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        x, y = p1

        # Define vertical goal box boundaries
        y_min = (self.pitch_config.width - self.pitch_config.goal_box_width) / 2
        y_max = (self.pitch_config.width + self.pitch_config.goal_box_width) / 2

        if self.verbose:
            print(f"Shot vector: vx={vx:.2f}, vy={vy:.2f} | y={y:.2f} | Y-Range: [{y_min:.2f}, {y_max:.2f}]")

        # Check if ball moves in correct direction and within vertical goal bounds
        if direction == "left_to_right":
            return vx > 50 and x > (self.pitch_config.length * 2 / 3) and y_min <= y <= y_max
        elif direction == "right_to_left":
            return vx < -50 and x < (self.pitch_config.length / 3) and y_min <= y <= y_max
        return False


    def _is_ball_in_goal_area(self, ball_x, ball_y, side: str) -> bool:
        """
        Returns True if ball is fully behind goal line AND within the goal mouth (between the posts).
        """
        # ⚽ Toleranz hinter Torlinie (X-Achse)
        goal_line_tolerance = 20  # cm

        # ⚽ Bereich zwischen Pfosten (Y-Achse) – realistisch mit kleiner Sicherheitsmarge
        y_center = self.pitch_config.width / 2
        goal_width = self.pitch_config.goal_width  # z. B. 732 cm (Standard-Torbreite)
        post_margin = 30  # cm Puffer an den Pfosten

        goal_y_min = y_center - goal_width / 2 + post_margin
        goal_y_max = y_center + goal_width / 2 - post_margin

        in_y_range = goal_y_min <= ball_y <= goal_y_max
        in_x_range = (
            ball_x < -goal_line_tolerance
            if side == "left"
            else ball_x > self.pitch_config.length + goal_line_tolerance
        )

        if self.verbose:
            print(
                f"[Goal Check] Side: {side} | X={ball_x:.1f}, Y={ball_y:.1f} | "
                f"in_x: {in_x_range}, in_y: {in_y_range} | "
                f"Y-Range: [{goal_y_min:.1f}, {goal_y_max:.1f}]"
            )

        return in_x_range and in_y_range

    def _get_team_for_goal_side(self, side: str):
        """
        Infers which team is attacking the goal on a given side.

        Args:
            side (str): 'left' or 'right' goal side.

        Returns:
            str or None: Team ID (as string) attacking that goal, or None if not found.
        """
        for team_id, direction in self.attack_directions.items():
            if (side == "left" and direction == "right_to_left") or (side == "right" and direction == "left_to_right"):
                return team_id
        return None

    def _register_goal(self, frame_idx: int, scoring_team: str, team_assignments: dict[int,int]):
        """
        Registers a goal event and determines scorer and assist based on shot history.

        Args:
            frame_idx (int): Frame where the goal occurred.
            scoring_team (str): ID of the team that scored.
        """
        if frame_idx in self.goals:
            return  # Goal already recorded for this frame

        print(f"Finding scorer for goal at frame {frame_idx}...")
        recent_shots = [s for s in self.shots if frame_idx - 150 <= s["frame"] <= frame_idx]
        print(f"Relevant shots: {recent_shots}")

        scorer, assist = None, None
        for i in range(len(recent_shots)-1, -1, -1):
            shot = recent_shots[i]
            shooter_id = shot["player"]
            if shooter_id is None or int(shooter_id) not in team_assignments:
                return  # Cannot proceed without shooter or team info

            shooter_team = int(team_assignments[int(shooter_id)])

            if str(shooter_team) == scoring_team:
                scorer = shooter_id
                for j in range(i - 1, -1, -1):
                    prev_shot = recent_shots[j]
                    prev_team = int(self.team_assignments.get("players", {}).get(prev_shot["player"], {}).get("team"))
                    if prev_team == scoring_team:
                        assist = prev_shot["player"]
                        break
                break

        # 🛑 Verwerfe das Tor, wenn kein Schütze gefunden wurde
        if scorer is None:
            print(f"⚠️ No scorer found for goal at frame {frame_idx} — ignoring goal.")
            return
        
        self.goals[frame_idx] = {
            "team": scoring_team,
            "scorer": scorer,
            "assist": assist
        }
        self.goal_counter[str(scoring_team)] += 1
        self.goal_cooldown_until_frame = frame_idx + self.goal_cooldown_frames

        print(f"Goal! Frame {frame_idx} | Team {scoring_team} | Scorer: {scorer} | Assist: {assist}")

    def _infer_shot_direction(self, frames_back=6):
        """
        Infers the shot direction by comparing current and past ball x-positions.

        Args:
            frames_back (int): How many frames to look back for direction estimation.

        Returns:
            str or None: 'left_to_right', 'right_to_left', or None if indeterminate.
        """
        if len(self.recent_ball_positions) < frames_back:
            return None
        x_now = self.recent_ball_positions[-1][0]
        x_then = self.recent_ball_positions[-frames_back][0]
        dx = x_now - x_then

        if dx > 0:
            return "left_to_right"
        elif dx < 0:
            return "right_to_left"
        return None

    def _is_ball_behind_goal_line(self, x, side: str) -> bool:
        """
        Checks if the ball has passed the goal line fully based on pitch side.

        Args:
            x (float): Ball x-position.
            side (str): 'left' or 'right'.

        Returns:
            bool: True if the ball is clearly behind the goal line.
        """
        if side == "left":
            return x < 0 - self.goal_depth_tolerance
        elif side == "right":
            return x > self.pitch_config.length + self.goal_depth_tolerance
        return False

    def get_goals(self):
        """
        Returns the dictionary of all registered goals.

        Returns:
            dict: Goals by frame with scorer and assist info.
        """
        return self.goals

    def get_goal_count(self):
        """
        Returns goal counters for each team.

        Returns:
            dict: Goal counts {"1": count, "2": count}
        """
        return self.goal_counter

    def get_shot_count_by_frame(self) -> Dict[int, Tuple[float, float]]:
        """
        Returns a dict of cumulative shot counts for each frame.

        Returns:
            Dict[int, Tuple[float, float]]: Frame index mapped to (team1_shots, team2_shots).
        """
        return self.framewise_shots

    def save(self, path):
        """
        Saves the goals and shot history as a JSON file.

        Args:
            path (str): File path to save the data.
        """
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return o

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Bestehende Datei lesen, falls vorhanden
        if os.path.exists(path):
            with open(path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # Goals und Shots aktualisieren
        existing_data["goals"] = self.goals

        # Konvertiere die shots-Liste in ein dict mit Frame als Key
        shots_dict = {
            str(shot["frame"]): {
                k: v for k, v in shot.items() if k != "frame"
            }
            for shot in self.shots
        }
        existing_data["shots"] = shots_dict

        # Speichern
        with open(path, "w") as f:
            json.dump(existing_data, f, indent=4, default=convert)
