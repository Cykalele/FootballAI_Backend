# 📁 core/utils/pipeline_runner.py

from concurrent.futures import ThreadPoolExecutor
import os
from fastapi.responses import JSONResponse
from core.tracking.VideoTracker import VideoTracker
from core.team_assignment.team_assignment import TeamAssignment
from core.annotators.VideoAnnotator import VideoAnnotator
from core.annotators.heatmap_annotator import HeatMapAnnotator
from config.settings import REID_MODEL_PATH
from core.view_transformation.transformer import ViewTransformer

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def run_video_tracking(video_path, paths, session_id):
    """
    Runs the player and ball tracking process using VideoTracker.
    Tracking results are stored in the specified tracking log path.
    """
    tracker = VideoTracker(
        video_path=video_path,
        reid_model_path=REID_MODEL_PATH,
        speed_tracking=False,
        force_track=True,
        tracking_output_path=paths["tracking_log"],
        session_id=session_id
    )
    tracker.track_video()

def run_Transformation(video_path, paths):
    """
    Runs the pitch view transformation process.
    Uses tracking data to transform object positions into a metric pitch coordinate system.
    """
    transformer = ViewTransformer()
    transformer.run(video_path, paths["tracking_log"])  

def run_team_assignment(video_path, paths, teams, session_id, run_manual_assignment):
    """
    Assigns players to teams based on tracking data.
    Can run automatically or with manual user intervention via UI.
    """
    assigner = TeamAssignment(
        tracking_log_path=paths["tracking_log"],
        video_path=video_path,
        assignment_output_path=paths["team_assignments"],
        team_config=teams,
        session_id=session_id,
        run_manual_assignment=run_manual_assignment
    )
    assigner.assign_teams()

def run_annotating_video(paths, teams, video_path, fps, session_id):
    """
    Creates an annotated game video with overlays such as player markers, ball position,
    and metric-based visualizations (e.g., possession, shots).
    """
    annotator = VideoAnnotator(
        tracking_log_path=paths["tracking_log"],
        assignment_path=paths["team_assignments"],
        output_video_path=paths["annotated_video"],
        metric_path=paths,
        team_config=teams,
        session_id=session_id
    )
    annotator.annotate_video_from_path(video_path)

def run_heatmap(heatmap_mode, paths, teams, player_id=None):
    """
    Generates heatmaps for teams, players, or the ball.
    Supports three modes:
        - "player": per-player minute heatmaps
        - "team":   aggregated heatmap for each team (parallelized)
        - "ball":   ball movement or danger zone heatmap
    """
    if heatmap_mode == "player":
        heatmap_gen = HeatMapAnnotator(
            mode="player",
            player_id=player_id,
            team_config=teams,
            tracking_log=paths["tracking_log"],
            team_path=paths["team_assignments"],
            heatmap_path=paths["heatmap_dir"]
        )
        heatmap_gen.generate_team_minute_heatmaps()

    elif heatmap_mode == "team":
        def run_for_team(team_id):
            heatmap_gen = HeatMapAnnotator(
                mode="team",
                team_config=teams,
                tracking_log=paths["tracking_log"],
                team_path=paths["team_assignments"],
                heatmap_path=paths["heatmap_dir"]
            )
            heatmap_gen.generate_team_heatmap_single(team_id)

        # Parallel execution for both teams
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_for_team, tid) for tid in [1, 2]]
            for f in futures:
                f.result()  # Ensure exceptions are raised if any

    elif heatmap_mode == "ball":
        heatmap_gen = HeatMapAnnotator(
            mode="ball",
            team_config=teams,
            tracking_log=paths["tracking_log"],
            team_path=paths["team_assignments"],
            heatmap_path=paths["heatmap_dir"]
        )
        heatmap_gen.generate_player_or_danger_heatmap()

def run_full_pipeline(video_path, fps, team1_name, team1_color, team2_name, team2_color,
                      paths, session_id, run_tracking=True, run_view_transformation=True,
                      run_automatic_assignment=True, run_manual_assignment=False,
                      run_annotating=False, heatmap_mode=None, player_id=None):
    """
    Orchestrates the complete video analysis pipeline.
    Steps (conditional execution based on parameters):
        1. Tracking of players and ball
        2. Pitch view transformation
        3. Team assignment (automatic or manual)
        4. Video annotation with overlays
        5. Heatmap generation for teams and ball
    """
    teams = {
        "1": {"name": team1_name, "color": team1_color},
        "2": {"name": team2_name, "color": team2_color}
    }

    if run_tracking:
        run_video_tracking(video_path=video_path, paths=paths, session_id=session_id)

    if run_view_transformation:
        run_Transformation(video_path, paths)      

    if not os.path.exists(paths["team_assignments"]) and (run_automatic_assignment or run_manual_assignment):
        run_team_assignment(video_path, paths, teams, session_id, run_manual_assignment)

    if run_annotating:
        run_annotating_video(paths, teams, video_path, fps, session_id)
        print("✅ Annotation launched asynchronously.")
        print(f"📊 Generating heatmap: {heatmap_mode}")
        run_heatmap("team", paths, teams, player_id)
        run_heatmap("ball", paths, teams, player_id)
        print("✅ Heatmap generated.")
