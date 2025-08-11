from typing import Dict, Tuple
import numpy as np
import cv2
import supervision as sv


class MetricAnnotator:
    """
    Annotates video frames with a scoreboard and a metrics box displaying
    match statistics such as passes, possession, and shots on target.
    """

    def __init__(self, team_config: dict):
        """
        Initializes the MetricAnnotator using a team configuration dictionary.

        Parameters
        ----------
        team_config : dict
            Dictionary containing team visual and textual configuration.
            Expected format:
            {
                "1": {"color": "#HEXCODE", "name": "Team A"},
                "2": {"color": "#HEXCODE", "name": "Team B"}
            }
        """
        self.team_config = team_config
        self.team1_color = sv.Color.from_hex(self.team_config["1"]["color"]).as_bgr()
        self.team2_color = sv.Color.from_hex(self.team_config["2"]["color"]).as_bgr()
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6

    def annotate_scoreboard(self, frame: np.ndarray, team1_score: int, team2_score: int) -> np.ndarray:
        """
        Draws a scoreboard in the top-left corner showing team colors, names, and the current score.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame to annotate.
        team1_score : int
            The score of team 1.
        team2_score : int
            The score of team 2.

        Returns
        -------
        np.ndarray
            Frame annotated with the scoreboard overlay.
        """
        x, y = 20, 20
        w, h = 240, 45

        # Draw translucent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.black, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Team 1 icon and name
        cv2.circle(frame, (x + 20, y + 22), 10, self.team1_color, -1)
        cv2.putText(frame, self.team_config["1"]["name"], (x + 35, y + 29),
                    self.font, self.font_scale, self.white, 1, cv2.LINE_AA)

        # Score in the center
        score_text = f"{team1_score} : {team2_score}"
        score_size = cv2.getTextSize(score_text, self.font, self.font_scale + 0.2, 2)[0]
        score_x = x + (w - score_size[0]) // 2
        cv2.putText(frame, score_text, (score_x, y + 30),
                    self.font, self.font_scale + 0.2, self.white, 2, cv2.LINE_AA)

        # Team 2 name and icon
        team2_text_size = cv2.getTextSize(self.team_config["2"]["name"], self.font, self.font_scale, 1)[0]
        team2_x = x + w - team2_text_size[0] - 35
        cv2.putText(frame, self.team_config["2"]["name"], (team2_x, y + 29),
                    self.font, self.font_scale, self.white, 1, cv2.LINE_AA)
        cv2.circle(frame, (x + w - 20, y + 22), 10, self.team2_color, -1)

        return frame

    def annotate_metrics_box(
        self,
        frame: np.ndarray,
        frame_idx: int,
        metrics: Dict[str, Dict[int, Tuple[float, float]]],
        goals: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Draws a metrics box in the lower-left corner with team statistics and progress bars.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame to annotate.
        frame_idx : int
            Index of the current frame, used to retrieve time-dependent metric values.
        metrics : dict
            Dictionary mapping metric names to framewise values (team1_value, team2_value).
        goals : tuple of int, optional
            The current score for both teams (team1_goals, team2_goals).

        Returns
        -------
        np.ndarray
            Frame annotated with scoreboard and metrics overlay.
        """
        # Draw scoreboard at the top first
        frame = self.annotate_scoreboard(frame, goals[0], goals[1])

        # Box layout parameters
        num_metrics = len(metrics)
        box_width, base_height = 300, 70
        spacing_between_metrics = 45
        spacing = 50
        box_height = base_height + num_metrics * spacing_between_metrics
        x, y = 20, frame.shape[0] - box_height - 50

        # Background for metrics box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Header section
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (x, y), (x + box_width, y + 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(header_overlay, 0.6, frame, 0.4, 0)

        # Header content: team icons, names, and "Stats" label
        cv2.circle(frame, (x + 25, y + 15), 10, self.team1_color, -1)
        cv2.putText(frame, self.team_config["1"]["name"], (x + 40, y + 22),
                    self.font, self.font_scale, self.white, 1, cv2.LINE_AA)

        cv2.circle(frame, (x + box_width - 25, y + 15), 10, self.team2_color, -1)
        cv2.putText(frame, self.team_config["2"]["name"], (x + box_width - 75, y + 22),
                    self.font, self.font_scale, self.white, 1, cv2.LINE_AA)

        cv2.putText(frame, "Stats", (x + box_width // 2 - 30, y + 22),
                    self.font, 0.7, self.white, 2, cv2.LINE_AA)

        # Parameters for bar drawing
        bar_width = int(box_width * 0.35)
        bar_height = 18

        # Draw each metric row
        for idx, (metric_name, framewise_data) in enumerate(metrics.items()):
            y_offset = y + 50 + idx * spacing
            bar_x = x + (box_width - bar_width) // 2
            bar_y = y_offset + 15

            team1_val, team2_val = framewise_data.get(frame_idx, (0.0, 0.0))

            # Determine widths for bars based on metric type
            if metric_name in ["Passes", "Shots on Target"]:
                total = team1_val + team2_val
                team1_ratio = team1_val / total if total > 0 else 0.5
                team1_width = int(bar_width * team1_ratio)
                team2_width = bar_width - team1_width
            else:
                team1_width = int(bar_width * (team1_val / 100))
                team2_width = bar_width - team1_width

            # Metric label
            text_size = cv2.getTextSize(metric_name, self.font, self.font_scale, 1)[0]
            text_x = x + (box_width - text_size[0]) // 2
            cv2.putText(frame, metric_name, (text_x, y_offset),
                        self.font, self.font_scale, self.white, 1, cv2.LINE_AA)

            # Bar segments for each team
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + team1_width, bar_y + bar_height), self.team1_color, -1)
            cv2.rectangle(frame, (bar_x + team1_width, bar_y),
                          (bar_x + team1_width + team2_width, bar_y + bar_height), self.team2_color, -1)

            # Numeric values near the bars
            if metric_name in ["Passes", "Shots on Target"]:
                cv2.putText(frame, f"{int(team1_val)}", (x + 40, bar_y + bar_height - 3),
                            self.font, self.font_scale, self.white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(team2_val)}", (x + box_width - 75, bar_y + bar_height - 3),
                            self.font, self.font_scale, self.white, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"{int(team1_val)}%", (x + 40, bar_y + bar_height - 3),
                            self.font, self.font_scale, self.white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(team2_val)}%", (x + box_width - 75, bar_y + bar_height - 3),
                            self.font, self.font_scale, self.white, 1, cv2.LINE_AA)

        return frame
