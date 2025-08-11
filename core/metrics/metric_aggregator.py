import json
import os
import numpy as np
import pandas as pd
from api.session_manager import get_output_paths

SESSION_ROOT = "./sessions"

class MetricAggregator:
    """
    Aggregates precomputed football analytics from a session folder into
    concise team and time based summaries without altering the underlying data.

    This class performs read only operations on JSON artifacts that were
    produced earlier in the pipeline, for example tracking, event detection,
    and spatial metrics. It caches loaded content to avoid repeated I/O and
    offers exports in tabular form for reporting.

    Parameters
    ----------
    session_id : str
        Identifier of the session whose artifacts should be aggregated.

    Attributes
    ----------
    paths : dict
        Resolved file system paths for all required artifacts of the session.
    *_summary, *_distance, *_speed, player_stats_df, spatial_* :
        Lazily populated caches that hold intermediate and final summaries.
    """

    def __init__(self, session_id):
        # Lazy caches. They remain None until their first use.
        self.team_assignments = None
        self.player_team_map = None
        self.ball_possession = None
        self.goals_summary = None
        self.shots_summary = None
        self.total_distance = None
        self.avg_speed = None
        self.player_stats_df = None
        self.spatial_metrics = None
        self.spatial_summary = None

        # Centralized resolution of session specific artifact paths.
        self.paths = get_output_paths(SESSION_ROOT, session_id)

    def _load_team_assignments(self):
        """
        Loads the team assignment JSON once and keeps it in memory.

        Returns
        -------
        dict
            Full content of the team assignment file as a dictionary.
        """
        if self.team_assignments is None:
            path = self.paths["team_assignments"]
            with open(path, "r", encoding="utf-8") as f:
                self.team_assignments = json.load(f)
        return self.team_assignments

    def _build_team_mapping(self):
        """
        Builds a mapping from player identifier to team identifier.

        Players that are flagged as removed or that do not belong to team 1
        or team 2 are excluded. Keys are strings because upstream JSON keys
        are strings.

        Returns
        -------
        dict
            Mapping 'player_id' -> team integer in {1, 2}.
        """
        if self.player_team_map is None:
            assignments = self._load_team_assignments()
            self.player_team_map = {
                str(pid): int(pdata["team"])
                for pid, pdata in assignments.get("players", {}).items()
                if not pdata.get("removed", False) and str(pdata.get("team")) in ["1", "2"]
            }

        return self.player_team_map

    def load_ball_possession(self):
        """
        Loads team level ball possession shares in percent.

        Returns
        -------
        dict
            Keys 'team_1_possession_percent' and 'team_2_possession_percent'.
        """
        if self.ball_possession is None:
            path = self.paths["ball_possession"]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.ball_possession = {
                    "team_1_possession_percent": data["summary"]["team_1_possession_percent"],
                    "team_2_possession_percent": data["summary"]["team_2_possession_percent"]
                }
        return self.ball_possession

    def load_goals_and_shots(self):
        """
        Loads goals and derives shot counts per team by resolving the shooting
        player to a team. Goals already carry a team label and are counted
        directly. Shots on target are counted via the 'on_goal' flag.

        Returns
        -------
        tuple(dict, dict)
            First dictionary contains goal counts per team.
            Second dictionary contains total shots and shots on target per team.
        """
        if self.goals_summary is None or self.shots_summary is None:
            path = self.paths["ball_events"]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            player_team_map = self._build_team_mapping()
            shots_dict = data.get("shots", {})

            # Initialize counters for total shots and shots on target.
            shot_counts = {
                "team_1_shots": 0,
                "team_2_shots": 0,
                "team_1_on_goal": 0,
                "team_2_on_goal": 0
            }
            for shot in shots_dict.values():
                player_id = str(shot.get("player"))
                team_id = player_team_map.get(player_id)
                if team_id == 1:
                    shot_counts["team_1_shots"] += 1
                    if shot.get("on_goal"):
                        shot_counts["team_1_on_goal"] += 1
                elif team_id == 2:
                    shot_counts["team_2_shots"] += 1
                    if shot.get("on_goal"):
                        shot_counts["team_2_on_goal"] += 1

            self.shots_summary = shot_counts

            # Goals are labeled with the scoring team and can be summed directly.
            goals_data = data.get("goals", {})
            team_goals = {1: 0, 2: 0}
            for goal in goals_data.values():
                team_id = int(goal.get("team"))
                if team_id in team_goals:
                    team_goals[team_id] += 1

            self.goals_summary = {
                "team_1_goals": team_goals[1],
                "team_2_goals": team_goals[2]
            }

        return self.goals_summary, self.shots_summary

    def load_distance_and_speed(self):
        """
        Aggregates total distance in meters and average speed in km/h per team.
        Per player statistics are also retained as a DataFrame for downstream
        export or inspection.

        The average speed per player is computed as the arithmetic mean of the
        per frame speed values provided in the input JSON.

        Returns
        -------
        tuple(dict, dict)
            Team level total distance and team level average speed.
        """
        if self.total_distance is None or self.avg_speed is None or self.player_stats_df is None:
            player_map = self._build_team_mapping()
            path = self.paths["player_metrics"]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            team_distances = {1: 0.0, 2: 0.0}
            team_speeds = {1: [], 2: []}
            player_rows = []

            for pid, pdata in data["players"].items():
                team = player_map.get(str(pid))
                if team in (1, 2):
                    dist = pdata.get("total_distance_m", 0.0)
                    speed_values = pdata.get("speeds_kmh", {}).values()
                    speed_list = list(speed_values)
                    avg_speed = round(np.mean(speed_list), 2) if speed_list else 0.0

                    team_distances[team] += dist
                    team_speeds[team].append(avg_speed)
                    player_rows.append({
                        "player_id": pid,
                        "team": team,
                        "distance_m": round(dist, 2),
                        "avg_speed_kmh": avg_speed
                    })

            self.total_distance = {
                "team_1_distance_m": round(team_distances[1], 2),
                "team_2_distance_m": round(team_distances[2], 2)
            }
            self.avg_speed = {
                "team_1_avg_speed_kmh": round(np.mean(team_speeds[1]), 2) if team_speeds[1] else 0.0,
                "team_2_avg_speed_kmh": round(np.mean(team_speeds[2]), 2) if team_speeds[2] else 0.0
            }
            self.player_stats_df = pd.DataFrame(player_rows)

        return self.total_distance, self.avg_speed

    def load_spatial_metrics(self):
        """
        Loads spatial control metrics and computes team averages across frames.

        The method expects per frame fields 'space_control' with two entries
        for the teams and 'thirds_control_percent' with zone level control for
        defensive, middle, and attacking thirds.

        Returns
        -------
        dict
            Averages of space control and thirds control per team with two
            decimals.
        """
        if self.spatial_summary is None:
            path = self.paths["spatial_metrics"]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            space_control_t1, space_control_t2 = [], []
            thirds = {
                "defensive": {1: [], 2: []},
                "middle": {1: [], 2: []},
                "attacking": {1: [], 2: []}
            }

            for frame_data in data.values():
                sc = frame_data["space_control"]
                space_control_t1.append(sc[0])
                space_control_t2.append(sc[1])
                thirds_data = frame_data["thirds_control_percent"]
                for zone in ["defensive", "middle", "attacking"]:
                    thirds[zone][1].append(thirds_data[zone]["1"])
                    thirds[zone][2].append(thirds_data[zone]["2"])

            self.spatial_summary = {
                "space_control_avg_team_1": round(np.mean(space_control_t1), 2),
                "space_control_avg_team_2": round(np.mean(space_control_t2), 2),
                "thirds_control_avg_defensive_team_1": round(np.mean(thirds["defensive"][1]), 2),
                "thirds_control_avg_defensive_team_2": round(np.mean(thirds["defensive"][2]), 2),
                "thirds_control_avg_middle_team_1": round(np.mean(thirds["middle"][1]), 2),
                "thirds_control_avg_middle_team_2": round(np.mean(thirds["middle"][2]), 2),
                "thirds_control_avg_attacking_team_1": round(np.mean(thirds["attacking"][1]), 2),
                "thirds_control_avg_attacking_team_2": round(np.mean(thirds["attacking"][2]), 2)
            }

        return self.spatial_summary

    def load_passes(self):
        """
        Loads total pass counts per team from the event log.

        Returns
        -------
        dict
            Keys 'team_1_passes' and 'team_2_passes'.
        """
        with open(self.paths["ball_events"], "r") as f:
            data = json.load(f)
            total_passes = data.get("passes", {}).get("total", {})
            return {
                "team_1_passes": total_passes.get("1", 0),
                "team_2_passes": total_passes.get("2", 0)
            }

    def get_metrics_summary(self):
        """
        Produces a single dictionary with all team level metrics combined.

        Returns
        -------
        dict
            Merged summary across possession, goals, shots, distance, speed,
            spatial control, and passes.
        """
        possession = self.load_ball_possession()
        goals, shots = self.load_goals_and_shots()
        distance, speed = self.load_distance_and_speed()
        spatial = self.load_spatial_metrics()
        passes = self.load_passes()  # Newly added metric

        return {
            **possession,
            **goals,
            **shots,
            **distance,
            **speed,
            **spatial,
            **passes
        }

    def _aggregate_distance_and_speed(self, player_metrics, player_map, frame_to_minute, stats):
        """
        Aggregates per player distance and speed into minute level buckets.

        Distance is computed from successive 2D positions and distributed
        uniformly over all minutes in which a player has positions recorded.
        Speed values are grouped by minute and averaged later.

        Parameters
        ----------
        player_metrics : dict
            Per player positions and speeds as produced upstream.
        player_map : dict
            Mapping from player id to team id.
        frame_to_minute : callable
            Function that converts a frame index to a minute index.
        stats : dict
            Accumulator keyed by (minute, player_id, team).
        """
        for pid, pdata in player_metrics.items():
            pid = str(pid)
            team = player_map.get(pid)
            if team is None:
                continue

            # Extract positions and speeds with integer frame keys.
            positions = {
                int(frame): np.array(pos) for frame, pos in pdata.get("positions", {}).items()
            }
            speeds = {
                int(frame): float(kmh) for frame, kmh in pdata.get("speeds_kmh", {}).items()
            }

            # Compute total distance from successive position vectors.
            sorted_positions = sorted(positions.items())
            position_vectors = [np.array(pos) for _, pos in sorted_positions]
            total_distance = sum(
                np.linalg.norm(p2 - p1)
                for p1, p2 in zip(position_vectors, position_vectors[1:])
            )

            # Group positions by minute to determine active minutes.
            minutes_frames = {}
            for frame, pos in positions.items():
                minute = frame_to_minute(frame)
                minutes_frames.setdefault(minute, []).append((frame, pos))

            # Group speeds by minute for later averaging.
            minutes_speeds = {}
            for frame, speed in speeds.items():
                minute = frame_to_minute(frame)
                minutes_speeds.setdefault(minute, []).append(speed)

            active_minutes = set(minutes_frames.keys())

            if not active_minutes:
                # If a player has no active minutes, initialize with zeros in the first observed minute.
                if positions:
                    first_frame = min(positions.keys())
                    minute = frame_to_minute(first_frame)
                    key = (minute, pid, team)
                    stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                continue

            # Distribute distance evenly across active minutes to obtain minute level distance.
            dist_per_minute = total_distance / len(active_minutes)

            for minute in active_minutes:
                key = (minute, pid, team)
                stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                stats[key]["distance"] = round(dist_per_minute, 2)

            # Append speeds for each minute for subsequent averaging.
            for minute, speed_list in minutes_speeds.items():
                key = (minute, pid, team)
                stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                stats[key]["speed"].extend(speed_list)

    def _aggregate_ball_possession(self, possession_data, player_map, frame_to_minute, stats):
        """
        Aggregates ball possession at the player minute level by counting the
        number of frames per minute in which a player is in possession.

        Parameters
        ----------
        possession_data : dict
            Frame indexed possession annotations with 'player_id'.
        """
        for frame_str, pdata in possession_data.items():
            frame = int(frame_str)
            player_id = pdata.get("player_id")
            if player_id:
                minute = frame_to_minute(frame)
                team = player_map.get(str(player_id))
                if team is not None:
                    key = (minute, str(player_id), team)
                    stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                    stats[key]["ball"] += 1

    def _aggregate_shots(self, shots, player_map, frame_to_minute, stats):
        """
        Aggregates shots at the player minute level.

        Parameters
        ----------
        shots : dict or list
            Shot events indexed by frame or provided as a list of events.
        """
        for frame_str, shot in shots.items():
            frame = int(frame_str)

            pid = str(shot["player"])
            minute = frame_to_minute(frame)
            team = player_map.get(pid)
            if team is not None:
                key = (minute, pid, team)
                stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                stats[key]["shots"] += 1
            print(f"[DEBUG] Shot by player {pid} in frame {frame} (minute {minute}), team {team}")

    def _aggregate_goals(self, goals, frame_to_minute, stats):
        """
        Aggregates goals at the team minute level since goals are team events.

        Parameters
        ----------
        goals : dict
            Goal events indexed by frame containing the team identifier.
        """
        for frame_str, goal in goals.items():
            frame = int(frame_str)
            minute = frame_to_minute(frame)
            team_id = int(goal["team"])

            key = (minute, f"goal_team_{team_id}", team_id)
            stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0, "goals": 0})
            stats[key]["goals"] += 1

    def _aggregate_passes(self, pass_data, player_map, frame_to_minute, stats):
        """
        Aggregates team level pass counts at minute resolution.

        The input contains cumulative pass counts per frame for both teams.
        For each minute the last observed frame is used to obtain the minute
        level count. Player level attribution is not attempted.

        Parameters
        ----------
        pass_data : dict
            Mapping 'frame' -> (team1_count, team2_count).
        """
        # Keep only the last frame per minute to avoid double counting.
        last_frame_per_minute = {}

        for frame_str, (team1, team2) in pass_data.items():
            frame = int(frame_str)
            minute = frame_to_minute(frame)
            last_frame_per_minute.setdefault(minute, (frame, (team1, team2)))
            if frame > last_frame_per_minute[minute][0]:
                last_frame_per_minute[minute] = (frame, (team1, team2))

        for minute, (_, (team1, team2)) in last_frame_per_minute.items():
            for team, count in [(1, team1), (2, team2)]:
                key = (minute, f"team{team}", team)
                stats.setdefault(key, {"distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0})
                stats[key]["passes"] = count

    def export_team_summary_metrics_excel(self, writer):
        """
        Writes a compact team summary sheet to an open Excel writer.

        The method harmonizes metric keys to human readable labels and produces
        a table with one row per metric and one column per team.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Open Excel writer with an active workbook context.
        """
        metrics_mapping = {
            "team_1_possession_percent": ("Ballbesitz (%)", 1),
            "team_2_possession_percent": ("Ballbesitz (%)", 2),
            "team_1_goals": ("Tore", 1),
            "team_2_goals": ("Tore", 2),
            "team_1_shots": ("Torschüsse", 1),
            "team_2_shots": ("Torschüsse", 2),
            "team_1_passes": ("Pässe", 1),
            "team_2_passes": ("Pässe", 2),
            "team_1_distance_m": ("Distanz (m)", 1),
            "team_2_distance_m": ("Distanz (m)", 2),
            "team_1_avg_speed_kmh": ("Ø Geschwindigkeit (km/h)", 1),
            "team_2_avg_speed_kmh": ("Ø Geschwindigkeit (km/h)", 2),
            "space_control_avg_team_1": ("Raumkontrolle gesamt (%)", 1),
            "space_control_avg_team_2": ("Raumkontrolle gesamt (%)", 2),
            "thirds_control_avg_defensive_team_1": ("Defensiv-Kontrolle (%)", 1),
            "thirds_control_avg_defensive_team_2": ("Defensiv-Kontrolle (%)", 2),
            "thirds_control_avg_middle_team_1": ("Mittelfeld-Kontrolle (%)", 1),
            "thirds_control_avg_middle_team_2": ("Mittelfeld-Kontrolle (%)", 2),
            "thirds_control_avg_attacking_team_1": ("Offensiv-Kontrolle (%)", 1),
            "thirds_control_avg_attacking_team_2": ("Offensiv-Kontrolle (%)", 2),
        }

        metrics = self.get_metrics_summary()
        reformatted = {}

        # Rearrange by label and team so that the output is presentation ready.
        for key, value in metrics.items():
            if key in metrics_mapping:
                metric_label, team = metrics_mapping[key]
                if metric_label not in reformatted:
                    reformatted[metric_label] = {1: None, 2: None}
                reformatted[metric_label][team] = value

        # Build the output table with one row per metric.
        rows = []
        for metric_label, values in reformatted.items():
            rows.append({
                "Metrik": metric_label,
                "Team 1": values.get(1),
                "Team 2": values.get(2)
            })

        df_summary = pd.DataFrame(rows).sort_values(by="Metrik")
        df_summary.to_excel(writer, sheet_name="Team Summary", index=False)

    def write_minute_team_metrics(self, writer):
        """
        Writes a per minute team sheet that aggregates distance, speed,
        possession, shots, goals, and passes.

        Frames are converted to minutes using a constant frame rate of thirty
        frames per second. If the original video has a different frame rate
        the conversion must be adjusted upstream or here accordingly.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Open Excel writer with an active workbook context.
        """
        player_map = self._build_team_mapping()

        with open(self.paths["player_metrics"], "r") as f:
            player_metrics = json.load(f)["players"]

        with open(self.paths["ball_possession"], "r") as f:
            data = json.load(f)
            possession_data = data["frames"]

        with open(self.paths["ball_events"], "r") as f:
            data = json.load(f)
            pass_data = data.get("passes", {}).get("framewise", {})
            shots = data.get("shots", [])
            goals = data.get("goals", {})

        # Fixed frame rate assumption. Adjust if sessions use a different fps.
        fps = 30
        frame_to_minute = lambda f: f // (60 * fps)
        stats = {}

        # Populate the accumulator with all minute level components.
        self._aggregate_distance_and_speed(player_metrics, player_map, frame_to_minute, stats)
        self._aggregate_ball_possession(possession_data, player_map, frame_to_minute, stats)
        self._aggregate_shots(shots, player_map, frame_to_minute, stats)
        self._aggregate_goals(goals, frame_to_minute, stats)
        self._aggregate_passes(pass_data, player_map, frame_to_minute, stats)

        # Aggregate to team minute level by summing player contributions and
        # collecting speeds for averaging.
        team_minute_agg = {}
        for (minute, _, team), values in stats.items():
            key = (minute, team)
            team_minute_agg.setdefault(key, {
                "distance": 0.0, "speed": [], "ball": 0, "shots": 0, "passes": 0, "goals": 0, "players": 0
            })

            team_minute_agg[key]["distance"] += values["distance"]
            team_minute_agg[key]["speed"].extend(values["speed"])
            team_minute_agg[key]["ball"] += values["ball"]
            team_minute_agg[key]["shots"] += values["shots"]
            team_minute_agg[key]["passes"] += values["passes"]
            team_minute_agg[key]["goals"] += values.get("goals", 0)
            team_minute_agg[key]["players"] += 1

        # Build the output table. Possession is converted from frames to percent.
        rows = []
        for (minute, team), values in team_minute_agg.items():
            ball_frames = values["ball"]
            ball_poss_percent = round(100 * ball_frames / (60 * fps), 2)
            avg_speed = round(np.mean(values["speed"]), 2) if values["speed"] else 0.0
            avg_distance = round(values["distance"], 2)
            rows.append({
                "Minute": minute,
                "Team": team,
                "Avg Distance (m)": avg_distance,
                "Avg Speed (km/h)": avg_speed,
                "Ball Possession (%)": ball_poss_percent,
                "Shots": values["shots"],
                "Goals": values["goals"],
                "Passes": values["passes"]
            })

        df = pd.DataFrame(rows).sort_values(by=["Minute", "Team"])
        df.to_excel(writer, sheet_name="Per Minute Team Metrics", index=False)

    def export_combined_metrics_excel(self, output_path):
        """
        Creates an Excel workbook with a team summary sheet and a per minute
        team sheet. The directory structure is created if it does not exist.

        Parameters
        ----------
        output_path : str
            Target path of the Excel file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            self.export_team_summary_metrics_excel(writer)
            self.write_minute_team_metrics(writer)
