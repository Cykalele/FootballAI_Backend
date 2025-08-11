import os
import numpy as np

from core.metrics.ball_posession import BallPossessionAnalyzer
from core.metrics.player_metrics import PlayerMetricsAnalyzer
from core.metrics.spatial_metrics import SpatialAnalyzer
from core.annotators.metric_annotator import MetricAnnotator
from core.metrics.detect_goal import GoalDetector  
from core.metrics.metric_aggregator import MetricAggregator

class MetricsAnalyzer:
    """
    Central orchestrator for computing, updating, and saving football-specific metrics per video frame.

    Responsibilities:
    - Ball possession analysis
    - Player movement metrics (distance, speed)
    - Spatial control metrics (Voronoi control areas)
    - Goal and shot detection
    - On-frame metric annotation for visualization
    """
    
    def __init__(self, metric_path, assignment_path, team_config, session_id):
        """
        Initializes the orchestrator and sets up all metric modules.

        Args:
            metric_path (dict): Dictionary with file paths for saving each metric category.
            assignment_path (str): Path to the team assignment JSON file.
            team_config (dict): Configuration for team visualization (e.g., colors, names).
            session_id (str): Unique identifier for the current video analysis session.
        """
        self.metric_path = metric_path
        self.ball_possession = BallPossessionAnalyzer(assignment_path)    # Tracks ball possession per frame
        self.player_metrics = PlayerMetricsAnalyzer()                     # Tracks distance and speed for players
        self.spatial_analyzer = SpatialAnalyzer(assignment_path)          # Computes spatial control (Voronoi areas)
        self.metric_annotator = MetricAnnotator(team_config)               # Draws metrics overlay on frames
        self.goal_detector = GoalDetector(assignment_path)                 # Detects goals and shots
        self.metric_aggregator = MetricAggregator(session_id=session_id)   # Aggregates metrics for summary and export

    def update(self, frame_idx: int, pitch_ball_coords: np.ndarray, pitch_player_coords_dict: dict[int, np.ndarray], player_ids: list[int], framewise_control_mask, frame_log, team_assignments):
        """
        Updates all metric modules with tracking data from a single frame.

        Args:
            frame_idx (int): Index of the current video frame.
            pitch_ball_coords (np.ndarray): Ball coordinates in pitch space (shape: 1x2).
            pitch_player_coords_dict (dict): Mapping from player ID to 2D coordinates in pitch space.
            player_ids (list[int]): List of player IDs visible in the current frame.
            framewise_control_mask (np.ndarray): Voronoi control mask for the frame.
            frame_log (dict): Raw tracking log for the current frame.
            team_assignments (dict): Mapping of player IDs to team information.
        """
        # Keep only players with valid pitch coordinates
        valid_entries = [
            (pid, pitch_player_coords_dict[int(pid)])
            for pid in player_ids
            if pid in pitch_player_coords_dict
        ]
        if valid_entries:
            valid_player_ids, valid_coords = zip(*valid_entries)
            player_coords_array = np.array(valid_coords, dtype=np.float32)
        else:
            valid_player_ids = []
            player_coords_array = np.empty((0, 2), dtype=np.float32)

        # Update ball possession status
        self.ball_possession.update(frame_idx, pitch_ball_coords, pitch_player_coords_dict, player_ids, team_assignments)

        # Update distance and speed metrics for players
        self.player_metrics.update(frame_idx, list(zip(valid_player_ids, player_coords_array)))

        # Update Voronoi-based spatial control
        self.spatial_analyzer.update(frame_idx=frame_idx, framewise_control_mask=framewise_control_mask)

        # Update goal and shot detection
        self.goal_detector.update(
            frame_idx=frame_idx,
            ball_pos=pitch_ball_coords[0] if isinstance(pitch_ball_coords, (list, np.ndarray)) and len(pitch_ball_coords) > 0 else None,
            player_positions=player_coords_array,
            ball_possession=self.ball_possession.get_possessor_by_frame(frame_idx),
            team_assignments=team_assignments
        )

    def annotate_overlay(self, frame, frame_idx):
        """
        Draws a metric overlay (KPIs like possession, space control, passes, goals) onto the frame.

        Args:
            frame (np.ndarray): The video frame to annotate.
            frame_idx (int): Index of the current frame.

        Returns:
            np.ndarray: Annotated frame with metric overlay.
        """
        # Retrieve ball possession per frame and ensure integer frame keys
        ball_pos = self.ball_possession.get_ball_posession()
        ball_pos_int_keys = {int(k): v for k, v in ball_pos.items()}

        # Collect all metrics to display
        metrics = {
            "Ball Possession": ball_pos_int_keys,
            "Space Control": self.spatial_analyzer.get_control_percentages(),
            "Passes": self.ball_possession.get_framewise_teamwise_pass_counts_by_frame(),
            "Shots on Target": self.goal_detector.get_shot_count_by_frame()
        }

        # Get team-wise goal counts
        goals = self.goal_detector.get_goal_count()
        goals_t1 = goals.get("1", 0)
        goals_t2 = goals.get("2", 0)

        # Annotate metrics onto the frame
        return self.metric_annotator.annotate_metrics_box(frame, frame_idx, metrics, goals=(goals_t1, goals_t2))

    def get_metrics_summary(self):
        """
        Retrieves aggregated metric summaries for the analyzed session.

        Returns:
            dict: Aggregated metrics.
        """
        return self.metric_aggregator.get_metrics_summary()

    def export_metrics_excel(self, metric_excel_path):
        """
        Exports all computed metrics into an Excel file.

        Args:
            metric_excel_path (str): Path to save the Excel file.
        """
        self.metric_aggregator.export_combined_metrics_excel(metric_excel_path)

    def save_all(self):
        """
        Saves all computed metrics to their respective file paths.
        """
        self.ball_possession.save(
            ball_possession_path=self.metric_path['ball_possession'],
            ball_events_path=self.metric_path['ball_events']
        )
        self.player_metrics.save(self.metric_path['player_metrics'])
        self.spatial_analyzer.save(self.metric_path['spatial_metrics'])
        self.goal_detector.save(self.metric_path['ball_events'])
        self.export_metrics_excel(self.metric_path['metrics_excel'])
