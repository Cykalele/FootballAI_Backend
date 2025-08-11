# 📁 Session Manager for managing sessions and their output structure
import os
import time
import uuid
from datetime import datetime
import psutil

# Root directories for session and metrics storage
SESSION_ROOT = "./sessions"
METRIC_ROOT = "./sessions"

# Ensure base directories exist
os.makedirs(SESSION_ROOT, exist_ok=True)
os.makedirs(METRIC_ROOT, exist_ok=True)


def get_output_paths(base_output_dir: str, session_id: str, create_dir: bool = False):
    """
    Build and return a dictionary containing all relevant file paths for a given session.
    Optionally creates the session directory.
    """
    session_dir = os.path.join(base_output_dir, session_id)
    if create_dir:
        os.makedirs(session_dir, exist_ok=True)

    return {
        "session_id": session_id,
        "session_dir": session_dir,
        "tracking_log": os.path.join(session_dir, "tracking_log.json"),
        "view_transform_flag": os.path.join(session_dir, "progress", "view_transformer_done.flag"),
        "team_assignments": os.path.join(session_dir, "team_assignments.json"),
        "annotated_video": os.path.join(session_dir, "annotated_game.mp4"),
        "ball_possession": os.path.join(session_dir, "ball_possession.json"),
        "player_metrics": os.path.join(session_dir, "player_metrics.json"),
        "spatial_metrics": os.path.join(session_dir, "spatial_metrics.json"),
        "ball_events": os.path.join(session_dir, "ball_events.json"),
        "heatmap_dir": os.path.join(session_dir, "heatmaps"),
        "metrics_excel": os.path.join(session_dir, "metrics_summary.xlsx"),
        "team_config": os.path.join(session_dir, "team_config.json")
    }


def create_new_session():
    """
    Create a unique session ID based on current timestamp and a short UUID suffix.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]


def list_active_ports():
    """
    Returns a dictionary of active listening ports with associated process names and PIDs.
    """
    ports = {}
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            conns = proc.connections()
            for conn in conns:
                if conn.status == psutil.CONN_LISTEN:
                    ports[conn.laddr.port] = {
                        'pid': proc.pid,
                        'name': proc.name()
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return ports


def format_status():
    """
    Creates a status string showing whether specific tracked ports are active or free.
    """
    ports = list_active_ports()
    tracked_ports = [8000, 8501, 4040]
    output = ["\n📊 Active processes on important ports:"]
    for p in tracked_ports:
        if p in ports:
            info = ports[p]
            output.append(f"✅ Port {p}: {info['name']} (PID {info['pid']})")
        else:
            output.append(f"❌ Port {p}: free")
    return "\n".join(output)


if __name__ == "__main__":
    # Create a new session and prepare output structure
    session_id = create_new_session()
    paths = get_output_paths(SESSION_ROOT, session_id, create_dir=True)

    print(f"🆕 New session created: {session_id} → {paths['session_dir']}")
    print(format_status())

    print("\n📁 Output structure:")
    for k, v in paths.items():
        print(f"{k}: {v}")
