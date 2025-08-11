import subprocess
from fastapi import APIRouter, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
import cv2
import json

from api.session_manager import get_output_paths
from api.services.pipeline_runner import run_full_pipeline
from api.services.video_utils import is_valid_mp4, robust_download_video
from config.settings import API_BASE_PATH

# Create API router with a defined base path for all routes
router = APIRouter(prefix=API_BASE_PATH)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SESSION_ROOT = "./sessions"


@router.get("/{session_id}")
def get_session_info(session_id: str):
    """
    Returns the current processing status and stored configuration for a given session.

    Parameters
    ----------
    session_id : str
        Unique identifier for the processing session.

    Returns
    -------
    JSONResponse
        JSON object containing:
            - progress flags for each pipeline step
            - current team configuration (if available)
            - session metadata
    """
    session_path = os.path.join(SESSION_ROOT, session_id)
    tracking_flag = os.path.join(session_path, "progress", "tracker_done.flag")
    view_transfomer_flag = os.path.join(session_path, "progress", "view_transformer_done.flag")
    assignment_path = os.path.join(session_path, "progress", "team_done.flag")
    assignment_notification_exists = os.path.exists(os.path.join(session_path, "team_assignment_notification.json"))
    done_flag = os.path.join(session_path, "progress", "annotation_done.flag")
    config_path = os.path.join(session_path, "team_config.json")

    response = {
        "session_id": session_id,
        "tracking_exists": os.path.exists(tracking_flag),
        "view_exists": os.path.exists(view_transfomer_flag),
        "assignment_notification_exists": assignment_notification_exists,
        "assign_exists": os.path.exists(assignment_path),
        "annotated_exists": os.path.exists(done_flag),
        "team_config": None
    }

    # Attempt to read stored team configuration
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                response["team_config"] = json.load(f)
        except Exception as e:
            response["team_config"] = {"error": str(e)}

    return JSONResponse(content=response)


@router.post("/{session_id}/video")
async def upload_and_run(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    team1_name: str = Form(...),
    team1_color: str = Form(...),
    team2_name: str = Form(...),
    team2_color: str = Form(...),
    run_tracking: bool = Form(True),
    run_automatic_assignment: bool = Form(True),
    run_manual_assignment: bool = Form(False),
    run_annotating: bool = Form(False),
    heatmap_mode: str = Form(None),
    player_id: str = Form(None),
):
    """
    Uploads a local MP4 file for a given session and triggers the processing pipeline asynchronously.

    Parameters
    ----------
    session_id : str
        Unique identifier for the processing session.
    file : UploadFile
        Uploaded video file (MP4).
    team1_name, team2_name : str
        Display names for both teams.
    team1_color, team2_color : str
        Color codes (HEX or predefined) for each team.
    run_tracking : bool
        Whether to perform player/ball tracking.
    run_automatic_assignment : bool
        Whether to auto-assign teams based on jersey detection.
    run_manual_assignment : bool
        Whether to open the manual assignment UI.
    run_annotating : bool
        Whether to run annotation steps immediately (overwritten to False here).
    heatmap_mode : str
        Optional heatmap generation mode.
    player_id : str
        Optional filter for a single player's heatmap.

    Returns
    -------
    JSONResponse
        Immediate confirmation with link to the team assignment interface.
    """
    if not session_id:
        return JSONResponse({"status": "error", "message": "SESSION_ID not set."}, status_code=500)
    run_annotating = False  # Ensure annotation step is not triggered immediately

    session_dir = os.path.join(SESSION_ROOT, session_id)
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=True)
    input_path = os.path.join(session_dir, "input.mp4")

    # Save uploaded video file to session folder
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Read FPS from video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Store team configuration as JSON
    team_config = {
        "1": {"name": team1_name, "color": team1_color},
        "2": {"name": team2_name, "color": team2_color}
    }
    with open(os.path.join(session_dir, "team_config.json"), "w") as f:
        json.dump(team_config, f, indent=4)

    # Schedule the full processing pipeline as a background task
    background_tasks.add_task(
        run_full_pipeline,
        video_path=input_path,
        fps=fps,
        team1_name=team1_name,
        team1_color=team1_color,
        team2_name=team2_name,
        team2_color=team2_color,
        paths=paths,
        session_id=session_id,
        run_tracking=run_tracking,
        run_view_transformation=True,
        run_automatic_assignment=run_automatic_assignment,
        run_manual_assignment=run_manual_assignment,
        run_annotating=False,
        heatmap_mode=heatmap_mode,
        player_id=player_id
    )

    return JSONResponse({
        "status": "success",
        "session_id": session_id,
        "assignment_url": f"{API_BASE}/TeamAssignment?session_id={session_id}"
    })


@router.post("/{session_id}/video-from-link")
async def process_video_from_link(session_id: str, request: Request, background_tasks: BackgroundTasks):
    """
    Downloads a video from a remote URL, compresses it if necessary, and triggers the processing pipeline.

    Parameters
    ----------
    session_id : str
        Unique identifier for the processing session.
    request : Request
        JSON body containing:
            - video_url : str (required)
            - team1_name, team2_name : str
            - team1_color, team2_color : str
            - run_tracking : bool
            - run_automatic_assignment : bool
            - run_manual_assignment : bool
            - heatmap_mode : str
            - player_id : str

    Returns
    -------
    JSONResponse
        Immediate status confirmation after scheduling background processing.
    """
    payload = await request.json()
    video_url = payload.get("video_url")
    team1_name = payload.get("team1_name")
    team1_color = payload.get("team1_color")
    team2_name = payload.get("team2_name")
    team2_color = payload.get("team2_color")
    run_tracking = payload.get("run_tracking", True)
    run_automatic_assignment = payload.get("run_automatic_assignment", True)
    run_annotating = payload.get("run_annotating", False)
    heatmap_mode = payload.get("heatmap_mode", None)
    player_id = payload.get("player_id", None)
    run_manual_assignment = payload.get("run_manual_assignment")

    if not video_url:
        return JSONResponse({"status": "error", "message": "No video URL provided"}, status_code=400)
    if not session_id:
        return JSONResponse({"status": "error", "message": "SESSION_ID not set."}, status_code=500)

    session_dir = os.path.join(SESSION_ROOT, session_id)
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=True)
    raw_video_path = os.path.join(session_dir, "raw_input.mp4")
    compressed_video_path = os.path.join(session_dir, "input.mp4")

    # Download the video robustly (yt_dlp or similar logic)
    try:
        success = robust_download_video(video_url, raw_video_path)
        if not success or not is_valid_mp4(raw_video_path):
            raise RuntimeError("Download failed or file is not a valid MP4.")
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Download error: {str(e)}"}, status_code=500)

    # Compress video to reduce file size and ensure standard encoding
    try:
        ffmpeg_command = [
            "ffmpeg", "-i", raw_video_path,
            "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
            "-y", compressed_video_path
        ]
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError:
        return JSONResponse({"status": "error", "message": "Video compression failed"}, status_code=500)

    # Read FPS from compressed video
    cap = cv2.VideoCapture(compressed_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Store team configuration
    team_config = {
        "1": {"name": team1_name, "color": team1_color},
        "2": {"name": team2_name, "color": team2_color}
    }
    with open(os.path.join(session_dir, "team_config.json"), "w") as f:
        json.dump(team_config, f, indent=4)

    # Schedule the pipeline in the background
    background_tasks.add_task(
        run_full_pipeline,
        video_path=compressed_video_path,
        fps=fps,
        team1_name=team1_name,
        team1_color=team1_color,
        team2_name=team2_name,
        team2_color=team2_color,
        paths=paths,
        session_id=session_id,
        run_tracking=run_tracking,
        run_view_transformation=True,
        run_automatic_assignment=run_automatic_assignment,
        run_manual_assignment=run_manual_assignment,
        run_annotating=False,
        heatmap_mode=heatmap_mode,
        player_id=player_id
    )

    return JSONResponse({"status": "success", "session_id": session_id})
