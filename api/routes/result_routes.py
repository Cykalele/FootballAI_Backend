# result_routes.py
import json
import zipfile
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import os
from core.metrics.metrics_analyzer import MetricsAnalyzer
from core.metrics.metric_aggregator import MetricAggregator
from api.session_manager import get_output_paths
from api.services.pipeline_runner import run_heatmap
from config.settings import SESSION_ROOT
from concurrent.futures import ThreadPoolExecutor
from config.settings import API_BASE_PATH

# API router with a configurable base path
router = APIRouter(prefix=API_BASE_PATH)
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def api_url(endpoint: str) -> str:
    """Builds a full API URL from the configured base URL and endpoint."""
    return f"{API_BASE.rstrip('/')}{API_BASE_PATH.rstrip('/')}/{endpoint.lstrip('/')}"

@router.get("/{session_id}/results/metrics/summary")
def get_metrics_summary(session_id: str):
    """
    Returns a JSON summary of all calculated metrics for a given session.
    Includes team metadata (names and colors) from the stored configuration.
    """
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=False)
    session_dir = paths["session_dir"]

    analyzer = MetricAggregator(session_id=session_id)
    summary = analyzer.get_metrics_summary()

    # Load team configuration for naming and color information
    team_config_path = os.path.join(session_dir, "team_config.json")
    if not os.path.exists(team_config_path):
        return JSONResponse({"error": "team_config.json not found"}, status_code=404)

    with open(team_config_path, "r", encoding="utf-8") as f:
        team_config = json.load(f)

    return {
        "metrics": summary,
        "team_1": {
            "name": team_config["1"]["name"],
            "color": team_config["1"]["color"]
        },
        "team_2": {
            "name": team_config["2"]["name"],
            "color": team_config["2"]["color"]
        }
    }

@router.get("/{session_id}/results/metrics/excel")
def download_metrics_excel(session_id: str):
    """
    Provides the metrics summary as a downloadable Excel file.
    If the file does not exist, it will be generated first.
    """
    session_dir = os.path.join("sessions", session_id)
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=False)
    output_path = paths["metrics_excel"]

    if not os.path.exists(session_dir):
        return JSONResponse({"status": "error", "message": "Session not found"}, status_code=404)

    if not os.path.exists(output_path):
        analyzer = MetricAggregator(session_id=session_id)
        analyzer.export_combined_metrics_excel(output_path)

    if not os.path.exists(output_path):
        return JSONResponse({"status": "error", "message": "Export failed"}, status_code=500)

    return FileResponse(output_path,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename="metrics_summary.xlsx")

@router.get("/{session_id}/results/video/")
def download_annotated_video(session_id: str):
    """
    Returns the annotated video of the game if it exists.
    """
    video_path = os.path.join("sessions", session_id, "annotated_game.mp4")
    if not os.path.exists(video_path):
        return JSONResponse({"status": "error", "message": "Video not found"}, status_code=404)
    return FileResponse(video_path, filename="annotated_game.mp4", media_type="video/mp4")

@router.post("/{session_id}/results/heatmaps/generate")
def generate_team_heatmaps(session_id: str):
    """
    Generates heatmaps for teams and the ball based on tracking data.
    Zips them for convenient download.
    """
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=False)
    session_dir = paths["session_dir"]
    heatmap_dir = paths["heatmap_dir"]

    # Load team configuration
    team_config_path = os.path.join(session_dir, "team_config.json")
    if not os.path.exists(team_config_path):
        return JSONResponse({"status": "error", "message": "Team config not found"}, status_code=404)

    try:
        with open(team_config_path, "r", encoding="utf-8") as f:
            team_config = json.load(f)

        # Generate heatmaps for both teams and the ball
        run_heatmap("team", paths, team_config)
        run_heatmap("ball", paths, team_config)
        print("✅ Heatmaps generated successfully.")

    except Exception as e:
        print(f"[ERROR] Heatmap generation failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    # Create a ZIP file with all heatmaps
    zip_path = os.path.join(session_dir, "heatmaps.zip")
    try:
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(heatmap_dir):
                full_path = os.path.join(heatmap_dir, file)
                zipf.write(full_path, arcname=file)
        print(f"✅ Heatmaps ZIP saved: {zip_path}")
    except Exception as e:
        print(f"[ERROR] ZIP creation failed: {e}")
        return JSONResponse({"status": "error", "message": f"ZIP failed: {e}"}, status_code=500)

    return FileResponse(zip_path, filename="heatmaps.zip", media_type="application/zip")

@router.get("/{session_id}/results/heatmaps/{team}")
def get_team_heatmap(session_id: str, team: str):
    """
    Returns the heatmap image for a specific team.
    """
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=False)
    heatmap_dir = paths["heatmap_dir"]

    file_path = os.path.join(heatmap_dir, f"heatmap_team_{team}.png")
    if not os.path.exists(file_path):
        return JSONResponse({"status": "error", "message": "Heatmap not found"}, status_code=404)
    return FileResponse(file_path, media_type="image/png")
