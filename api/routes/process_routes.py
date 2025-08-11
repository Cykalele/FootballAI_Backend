from pathlib import Path
from fastapi import APIRouter, Request, Form, BackgroundTasks
import requests
from fastapi.responses import JSONResponse
import os
import json
import cv2
from api.session_manager import get_output_paths
from api.services.pipeline_runner import run_full_pipeline
from config.settings import API_BASE_PATH

# Create a FastAPI router with a prefix for all endpoints
router = APIRouter(prefix=API_BASE_PATH)

# Base URL for internal API calls
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def api_url(endpoint: str) -> str:
    """Builds a full API endpoint URL."""
    return f"{API_BASE.rstrip('/')}{API_BASE_PATH.rstrip('/')}/{endpoint.lstrip('/')}"

SESSION_ROOT = "./sessions"

# Cache to store last known progress states (avoids missing/empty progress updates)
last_known_progress = {}


@router.post("/{session_id}/process")
async def process_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    team1_name: str = Form(...),
    team1_color: str = Form(...),
    team2_name: str = Form(...),
    team2_color: str = Form(...),
    run_automatic_assignment: bool = Form(...),
    run_manual_assignment: bool = Form(...),
):
    """
    Main endpoint to start the processing pipeline for a given session.
    Runs in background to avoid blocking the request.
    Executes in three steps:
      1. View Transformation
      2. Team Assignment
      3. Annotation
    """

    # Get paths for all relevant session files
    paths = get_output_paths(SESSION_ROOT, session_id)
    input_path = os.path.join(SESSION_ROOT, session_id, "input.mp4")

    # Lock file to prevent multiple annotation processes from running in parallel
    lock_file = Path(SESSION_ROOT) / session_id / "progress" / "annotation.lock"

    # Prevent duplicate runs if lock already exists
    if lock_file.exists():
        return JSONResponse({"status": "error", "message": "Annotation already in progress."}, status_code=409)

    # Create lock
    lock_file.touch()

    # Pre-check: tracking log must exist
    if not os.path.exists(paths["tracking_log"]):
        lock_file.unlink()
        return JSONResponse({"status": "error", "message": "Tracking log missing."}, status_code=400)

    # Pre-check: input video must exist
    if not os.path.exists(input_path):
        lock_file.unlink()
        return JSONResponse({"status": "error", "message": "Input video missing."}, status_code=400)

    # Get FPS from input video (used for downstream processing)
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    def stepwise_annotation():
        """
        Background job that runs the annotation process in steps.
        Automatically triggers the next step if previous outputs are missing.
        """
        try:
            # Step 1: View Transformation
            if not os.path.exists(paths["view_transform_flag"]):
                print("🧭 Starting View Transformation...")
                run_full_pipeline(
                    input_path, fps, team1_name, team1_color, team2_name, team2_color,
                    paths, session_id,
                    run_tracking=False,
                    run_view_transformation=True,
                    run_automatic_assignment=run_automatic_assignment,
                    run_manual_assignment=run_manual_assignment,
                    run_annotating=False
                )
                # Trigger Step 2 by re-calling this endpoint with updated parameters
                requests.post(api_url(f"{session_id}/process"), data={
                    "team1_name": team1_name,
                    "team1_color": team1_color,
                    "team2_name": team2_name,
                    "team2_color": team2_color,
                    "run_automatic_assignment": False,
                    "run_manual_assignment": True,
                })
                return

            # Step 2: Team Assignment
            team_lock_file = Path(SESSION_ROOT) / session_id / "progress" / "team_assignment.lock"
            if team_lock_file.exists():
                print("⏳ Team assignment is locked – skipping.")
            else:
                if not os.path.exists(paths["team_assignments"]):
                    print("🏷️ Starting Team Assignment...")
                    team_lock_file.touch()  # Create lock for this step
                    try:
                        run_full_pipeline(
                            input_path, fps, team1_name, team1_color, team2_name, team2_color,
                            paths, session_id,
                            run_tracking=False,
                            run_view_transformation=False,
                            run_automatic_assignment=run_automatic_assignment,
                            run_manual_assignment=run_manual_assignment,
                            run_annotating=False
                        )

                        # Wait for team assignment output (up to 10 seconds)
                        import time
                        wait_time = 0
                        while wait_time < 10:
                            if os.path.exists(paths["team_assignments"]):
                                print("✅ Team assignment saved.")
                                # Trigger Step 3
                                requests.post(api_url(f"{session_id}/process"), data={
                                    "team1_name": team1_name,
                                    "team1_color": team1_color,
                                    "team2_name": team2_name,
                                    "team2_color": team2_color,
                                    "run_automatic_assignment": False,
                                    "run_manual_assignment": True,
                                })
                                return
                            time.sleep(0.5)
                            wait_time += 0.5

                        print("⚠️ team_assignments.json was not created in time.")
                    finally:
                        # Remove lock for team assignment
                        if team_lock_file.exists():
                            team_lock_file.unlink()
                else:
                    print("✅ Team assignment already exists – skipping Step 2.")

            # Step 3: Annotation
            if os.path.exists(paths["team_assignments"]):
                print("🎬 Starting Annotation...")
                run_full_pipeline(
                    input_path, fps, team1_name, team1_color, team2_name, team2_color,
                    paths, session_id,
                    run_tracking=False,
                    run_view_transformation=False,
                    run_automatic_assignment=True,
                    run_manual_assignment=True,
                    run_annotating=True
                )
            else:
                print("Annotation skipped – manual team assignment required.")
        finally:
            # Always remove the annotation lock when done
            if lock_file.exists():
                lock_file.unlink()

    # Run the annotation job in the background
    background_tasks.add_task(stepwise_annotation)
    return JSONResponse({"status": "success", "message": "Processing started in background."})


@router.get("/{session_id}/progress/{mode}")
def get_progress(mode: str, session_id: str):
    """
    Returns the progress for a given processing mode (e.g., tracking, annotation).
    Uses last known cached progress if file is not yet readable.
    """
    progress_path = os.path.join(SESSION_ROOT, session_id, "progress", f"{mode}_progress.json")

    # If no progress file exists, return a default value
    if not os.path.exists(progress_path):
        last_known_progress[session_id] = {"current": 0, "total": 1}
        return JSONResponse(content=last_known_progress[session_id])

    try:
        with open(progress_path, "r") as f:
            data = json.load(f)
            last_known_progress[session_id] = data
            return JSONResponse(content=data)

    except (json.JSONDecodeError, OSError) as e:
        # If file is incomplete or unreadable, return cached progress instead
        print(f"[WARN] {mode}_progress.json for {session_id} not readable: {e}")
        fallback = last_known_progress.get(session_id, {"current": 0, "total": 1})
        return JSONResponse(content=fallback)
