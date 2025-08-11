from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from config.settings import API_BASE_PATH
import os
import json
import base64
from pathlib import Path
import cv2
import requests
from config.settings import SESSION_ROOT

# Create an API router with a configured prefix
router = APIRouter(prefix=API_BASE_PATH)

# Read base API URL from environment or fallback to localhost
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def api_url(endpoint: str) -> str:
    """
    Build a complete API URL for internal calls.
    Ensures no duplicate slashes in the final URL.
    """
    return f"{API_BASE.rstrip('/')}{API_BASE_PATH.rstrip('/')}/{endpoint.lstrip('/')}"

@router.post("/{session_id}/team-assignment/notify")
async def notify_team_assignment(session_id: str, request: Request):
    """
    Receives a JSON notification containing the initial
    team assignment information from the frontend.
    Saves the payload to 'team_assignment_notification.json'
    inside the session directory.
    """
    data = await request.json()
    os.makedirs(f'sessions/{session_id}', exist_ok=True)

    with open(f'sessions/{session_id}/team_assignment_notification.json", "w') as f:
        json.dump(data, f, indent=4)

    return {"status": "ok"}


@router.get("/{session_id}/team-assignment/frames")
def get_assignment_frames(session_id: str):
    """
    Retrieves all assignment frames and related metadata
    for the team assignment UI.
    Combines:
      - Frame images from 'assignment_frames'
      - Player bounding boxes from tracking_log.json
      - Current team assignments from team_assignment_notification.json
      - Team configuration from team_config.json
    Returns a base64-encoded image and assigned player data per frame.
    """
    frame_dir = os.path.join(SESSION_ROOT, session_id, "assignment_frames")
    tracking_log_path = os.path.join(SESSION_ROOT, session_id, "tracking_log.json")
    team_config_path = os.path.join(SESSION_ROOT, session_id, "team_config.json")
    team_assignment_path = os.path.join(SESSION_ROOT, session_id, "team_assignment_notification.json")

    # Validate required files
    if not os.path.exists(frame_dir):
        return JSONResponse({"error": "Frame directory not found"}, status_code=404)
    if not os.path.exists(tracking_log_path):
        return JSONResponse({"error": "Tracking log not found"}, status_code=404)
    if not os.path.exists(team_config_path):
        return JSONResponse({"error": "team_config.json not found"}, status_code=404)
    if not os.path.exists(team_assignment_path):
        return JSONResponse({"error": "team_assignments.json not found"}, status_code=404)

    # Load tracking and configuration data
    with open(tracking_log_path, "r", encoding="utf-8") as f:
        tracking_data = json.load(f)
    with open(team_config_path, "r", encoding="utf-8") as f:
        team_config = json.load(f)
    with open(team_assignment_path, "r", encoding="utf-8") as f:
        team_assignment = json.load(f).get("players", {})

    # Map frame IDs to player data with team assignment
    frame_to_players = {}
    for frame_id_str, data in tracking_data.items():
        frame_key = f"frame_{int(frame_id_str):04d}"
        players = []
        for p in data.get("players", []):
            pid = str(p["id"])
            team_id = team_assignment.get(pid, {}).get("team", None)
            players.append({
                "id": pid,
                "bbox": p["bbox"],
                "team": team_id
            })
        frame_to_players[frame_key] = players

    # Build the result payload: images + player data
    result = []
    for filename in sorted(os.listdir(frame_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(frame_dir, filename)
        frame_id = Path(filename).stem
        with open(image_path, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        result.append({
            "frame_id": frame_id,
            "image": img_base64,
            "players": frame_to_players.get(frame_id, [])
        })

    return {"frames": result, "team_config": team_config}


@router.post("/{session_id}/team-assignment/save-manual")
async def save_team_assignment(session_id: str, request: Request, background_tasks: BackgroundTasks):
    """
    Saves a manually edited team assignment from the frontend.
    Merges with existing assignments if present and updates team IDs.
    Also triggers downstream annotation processing for the video.
    """
    try:
        data = await request.json()
        session_path = os.path.join(SESSION_ROOT, session_id)
        os.makedirs(session_path, exist_ok=True)

        assignment_path = os.path.join(session_path, "team_assignments.json")

        # Merge with existing assignments if available
        if os.path.exists(assignment_path):
            with open(assignment_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            updated_players = {
                int(pid): pdata for pid, pdata in data.get("players", {}).items()
            }
            existing_data["players"] = {
                int(pid): pdata for pid, pdata in existing_data.get("players", {}).items()
            }
            existing_data["players"].update(updated_players)

            # Ensure team value is an integer
            for pid, pdata in existing_data["players"].items():
                if isinstance(pdata, dict) and "team" in pdata:
                    try:
                        pdata["team"] = int(pdata["team"])
                    except (ValueError, TypeError):
                        pdata["team"] = -1
            existing_data["directions"] = data.get("directions", existing_data.get("directions", {}))

            with open(assignment_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(assignment_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        # Validate that input video exists
        input_path = os.path.join(session_path, "input.mp4")
        if not os.path.exists(input_path):
            return JSONResponse({"status": "error", "message": "Input video not found."}, status_code=404)

        # Read FPS from the video (used in annotation)
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Load team configuration for annotation
        with open(os.path.join(session_path, "team_config.json"), "r") as f:
            teams = json.load(f)

        annotate_payload = {
            "team1_name": teams["1"]["name"],
            "team1_color": teams["1"]["color"],
            "team2_name": teams["2"]["name"],
            "team2_color": teams["2"]["color"],
            "run_automatic_assignment": True,
            "run_manual_assignment": False,
        }

        # Mark progress
        flag_path = os.path.join(session_path, "progress", "team_done.flag")
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as flag_file:
            flag_file.write("done")

        # Trigger annotation via local API call
        try:
            r = requests.post(api_url(f"{session_id}/process"), data=annotate_payload, timeout=3)
            if r.status_code != 200:
                print(f"[WARN] Annotation trigger failed: {r.text}")
        except Exception as e:
            print(f"[WARN] Could not trigger annotation: {e}")

        return JSONResponse({
            "status": "success",
            "message": "Team assignment saved and annotation started.",
            "team_config": teams,
            "directions": data.get("directions", {})
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/{session_id}/team-assignment")
async def auto_save_team_assignment(session_id: str, request: Request):
    """
    Saves an automatically generated team assignment without manual intervention.
    Requires both 'players' and 'directions' keys in the input JSON.
    Triggers annotation immediately after saving.
    """
    try:
        data = await request.json()
        if "players" not in data or "directions" not in data:
            return JSONResponse({
                "status": "error",
                "message": "JSON must contain 'players' and 'directions' keys."
            }, status_code=400)

        session_path = os.path.join(SESSION_ROOT, session_id)
        os.makedirs(session_path, exist_ok=True)

        assignment_path = os.path.join(session_path, "team_assignments.json")
        with open(assignment_path, "w", encoding="utf-8") as f:
            json.dump({
                "players": data["players"],
                "directions": data["directions"]
            }, f, indent=4)

        # Ensure input video exists
        input_path = os.path.join(session_path, "input.mp4")
        if not os.path.exists(input_path):
            return JSONResponse({"status": "error", "message": "Input video not found."}, status_code=404)

        # Ensure team configuration exists
        team_config_path = os.path.join(session_path, "team_config.json")
        if not os.path.exists(team_config_path):
            return JSONResponse({"status": "error", "message": "team_config.json missing."}, status_code=400)

        with open(team_config_path, "r", encoding="utf-8") as f:
            teams = json.load(f)

        annotate_payload = {
            "team1_name": teams["1"]["name"],
            "team1_color": teams["1"]["color"],
            "team2_name": teams["2"]["name"],
            "team2_color": teams["2"]["color"],
            "run_automatic_assignment": True,
            "run_manual_assignment": False,
        }

        flag_path = os.path.join(session_path, "progress", "team_done.flag")
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as flag_file:
            flag_file.write("done")

        try:
            r = requests.post(api_url(f"{session_id}/process"), data=annotate_payload, timeout=20)
            if r.status_code != 200:
                print(f"[WARN] Annotation trigger failed: {r.status_code} → {r.text}")
        except Exception as e:
            print(f"[WARN] Could not trigger annotation: {e}")

        return JSONResponse({
            "status": "success",
            "message": "Automatic team assignment saved and annotation started.",
            "team_config": teams,
            "directions": data.get("directions", {})
        })

    except Exception as e:
        print(f"[ERROR] auto_save_team_assignment: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/{session_id}/team-assignment/directions")
def get_team_directions(session_id: str):
    """
    Retrieves the stored match directions for both teams
    from the 'team_assignments.json' file in the session folder.
    """
    path = os.path.join(SESSION_ROOT, session_id, "team_assignments.json")

    if not os.path.exists(path):
        return JSONResponse(
            {"status": "error", "message": "team_assignment.json not found"},
            status_code=404
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            directions = data.get("directions", {})
            dir1 = directions.get("1", "unknown")
            dir2 = directions.get("2", "unknown")

        return {
            "status": "success",
            "team_1_direction": dir1,
            "team_2_direction": dir2
        }

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )
