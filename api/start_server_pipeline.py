import subprocess
import os
import socket
import psutil
import time
import re
import threading
import tkinter as tk
from tkinter import messagebox, filedialog

import requests
from session_manager import create_new_session, format_status
import queue

# === Settings ===
# Default Python interpreter paths for backend and frontend virtual environments
DEFAULT_BACKEND_PYTHON = r"./footballAI_env/Scripts/python.exe"
DEFAULT_FRONTEND_PYTHON = r"../FootballAI_Frontend/football_frontend/Scripts/python.exe"

# Compute absolute base directory path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH_UI = os.path.join(BASE_DIR, "FootballAI_Frontend", "config.py")
DEFAULT_CONFIG_PATH_API = os.path.join(BASE_DIR, "FootballAI_Backend", "config", "settings.py")
DEFAULT_APP_PATH = os.path.join(BASE_DIR, "FootballAI_Frontend", "app.py")
GIT_UI_REPO = "https://github.com/ToHauser/FootballAI-UI.git"

# Default ports for backend (FastAPI) and frontend (Streamlit)
PORT = 8000
STREAMLIT_PORT = 8501

# Process handles for started services
backend_proc = None
streamlit_proc = None
cloudflare_proc = None

# Paths that can be modified through the config GUI
BACKEND_PYTHON = DEFAULT_BACKEND_PYTHON
FRONTEND_PYTHON = DEFAULT_FRONTEND_PYTHON
APP_PATH = DEFAULT_APP_PATH
CONFIG_PATH_UI = DEFAULT_CONFIG_PATH_UI
CONFIG_PATH_API = DEFAULT_CONFIG_PATH_API

# Queue to pass error messages from threads to GUI
error_queue = queue.Queue()


def check_for_errors_in_gui(app, messagebox):
    """
    Periodically checks for error messages from worker threads and shows them in a popup.
    """
    try:
        while True:
            err = error_queue.get_nowait()
            messagebox.showerror("❌ Error in server thread", err)
    except queue.Empty:
        pass
    app.after(500, check_for_errors_in_gui, app, messagebox)


def build_streamlit_component():
    """
    Builds the custom Streamlit React component if it has not been built yet.
    Runs npm install and npm run build inside the component directory.
    """
    build_dir = os.path.abspath(
        os.path.join(BASE_DIR, "FootballAI_Frontend", "streamlit_team_component", "template", "my_component", "frontend")
    )
    if not os.path.exists(os.path.join(build_dir, "build")):
        print("🛠️ Building Streamlit component with npm...")
        try:
            subprocess.run(["npm", "install"], cwd=build_dir, check=True)
            subprocess.run(["npm", "run", "build"], cwd=build_dir, check=True)
            print("✅ Build successful.")
        except subprocess.CalledProcessError as e:
            print("❌ Error building component:", e)


def is_port_in_use(port):
    """
    Checks if a given TCP port is currently in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def kill_process_on_port(port):
    """
    Kills any process currently listening on the given port.
    """
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    proc.kill()
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def ensure_venv_exists():
    """
    Ensures that the backend Python virtual environment exists.
    If not, creates it and installs dependencies from requirements.txt.
    """
    if not os.path.exists("footballAI_env"):
        print("📦 Creating virtual environment...")
        subprocess.run(["python", "-m", "venv", "footballAI_env"])
        subprocess.run(["footballAI_env/Scripts/pip", "install", "-r", "requirements.txt"])


def start_cloudflare_tunnel(port):
    """
    Starts a Cloudflare tunnel to expose the backend publicly.
    Returns the generated public URL.
    """
    global cloudflare_proc
    cloudflare_proc = subprocess.Popen(
        ["cloudflared.exe", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )

    for line in iter(cloudflare_proc.stdout.readline, ""):
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if match:
            return match.group(0)
    return None


def write_secrets_toml(api_base_url):
    """
    Writes the public API base URL into Streamlit's secrets.toml so the frontend can access it.
    """
    secrets_path = os.path.join(BASE_DIR, "FootballAI_Frontend", ".streamlit", "secrets.toml")
    os.makedirs(os.path.dirname(secrets_path), exist_ok=True)

    with open(secrets_path, "w", encoding="utf-8") as f:
        f.write(f'API_BASE = "{api_base_url}"\n')

    print(f"📄 secrets.toml written: {secrets_path}")


def print_backend_logs(proc):
    """
    Starts a thread that continuously reads and prints backend log output.
    """
    def log_reader():
        for line in iter(proc.stdout.readline, ""):
            print("[BACKEND]", line.strip())
    threading.Thread(target=log_reader, daemon=True).start()


def start_backend():
    """
    Starts the backend FastAPI server with uvicorn on the configured port.
    """
    global backend_proc
    backend_proc = subprocess.Popen(
        [BACKEND_PYTHON, "-m", "uvicorn", "main:app", "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )
    print_backend_logs(backend_proc)


def wait_for_backend(port, timeout=60):
    """
    Waits until the backend server responds or until a timeout occurs.
    """
    url = f"http://localhost:{port}/docs"
    start = time.time()
    print("⏳ Waiting for backend...")
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("✅ Backend is reachable.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError("❌ Backend not reachable within timeout.")


def start_streamlit(tunnel_url, session_id):
    """
    Starts the Streamlit frontend, passing the API base URL and session ID as environment variables.
    """
    global streamlit_proc
    os.environ["API_BASE"] = tunnel_url
    os.environ["SESSION_ID"] = session_id

    streamlit_proc = subprocess.Popen(
        [FRONTEND_PYTHON, "-m", "streamlit", "run", APP_PATH],
        cwd=os.path.dirname(APP_PATH)
    )


def stop_all():
    """
    Terminates all started processes: backend, frontend, and Cloudflare tunnel.
    """
    for proc in [backend_proc, streamlit_proc, cloudflare_proc]:
        if proc and proc.poll() is None:
            proc.terminate()


def git_push_ui_repo():
    """
    Automatically commits and pushes changes in the UI repository to GitHub.
    """
    try:
        repo_path = os.path.join(BASE_DIR, "FootballAI_Frontend")
        subprocess.run(["git", "add", "-A"], cwd=repo_path)
        subprocess.run(["git", "commit", "-m", "Automatischer Commit beim Start"], cwd=repo_path)
        subprocess.run(["git", "push"], cwd=repo_path)
    except Exception as e:
        print(f"Error during Git push: {e}")


def run_server(mode):
    """
    Starts the complete pipeline: backend, Cloudflare tunnel (if public mode), and frontend.
    """
    ensure_venv_exists()
    build_streamlit_component()

    # Free occupied ports
    if is_port_in_use(PORT):
        kill_process_on_port(PORT)
    if is_port_in_use(STREAMLIT_PORT):
        kill_process_on_port(STREAMLIT_PORT)

    # Create new analysis session
    session_id = create_new_session()

    # Start backend and wait for availability
    start_backend()
    wait_for_backend(PORT)

    if mode == "public":
        cf_url = start_cloudflare_tunnel(PORT)
        if cf_url:
            write_secrets_toml(cf_url)
            start_streamlit(cf_url, session_id)
            time.sleep(5)
            git_push_ui_repo()
        else:
            messagebox.showerror("Error", "Cloudflare tunnel could not be started.")
    else:
        local_url = f"http://localhost:{PORT}"
        start_streamlit(local_url, session_id)


def run_in_thread(mode):
    """
    Runs the server startup in a separate thread to keep the GUI responsive.
    """
    def safe_run():
        try:
            run_server(mode)
        except Exception as e:
            error_queue.put(str(e))
    threading.Thread(target=safe_run, daemon=True).start()


def open_config_window():
    """
    Opens a configuration window to change paths for backend/frontend executables and config files.
    """
    config_win = tk.Toplevel(app)
    config_win.title("🗂 Configure Paths")
    config_win.geometry("750x300")
    config_win.configure(bg="#1e1e1e")
    config_win.grab_set()

    path_entries = {}
    path_defaults = {
        "BACKEND_PYTHON": BACKEND_PYTHON,
        "FRONTEND_PYTHON": FRONTEND_PYTHON,
        "APP_PATH": APP_PATH,
        "CONFIG_PATH_UI": CONFIG_PATH_UI
    }

    path_labels = {
        "BACKEND_PYTHON": "⚙️ Backend Python Path",
        "FRONTEND_PYTHON": "🖼️ Frontend Python Path",
        "APP_PATH": "📄 Frontend app.py Path",
        "CONFIG_PATH_UI": "⚙️ Frontend config.py Path"
    }

    def create_path_row(row, var_name, label_text):
        tk.Label(config_win, text=label_text, bg="#1e1e1e", fg="white", font=("Segoe UI", 10)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        entry = tk.Entry(config_win, width=60, font=("Segoe UI", 10))
        entry.grid(row=row, column=1, padx=5)
        entry.insert(0, path_defaults.get(var_name, ""))
        path_entries[var_name] = entry

        def browse():
            path = filedialog.askopenfilename(parent=config_win, title=f"Select {label_text}")
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        tk.Button(config_win, text="📂", command=browse, bg="#eeeeee", fg="black", width=2).grid(row=row, column=2, padx=5)

    for i, (var_name, label_text) in enumerate(path_labels.items()):
        create_path_row(i, var_name, label_text)

    def save_paths():
        global BACKEND_PYTHON, FRONTEND_PYTHON, APP_PATH, CONFIG_PATH_UI
        BACKEND_PYTHON = path_entries["BACKEND_PYTHON"].get()
        FRONTEND_PYTHON = path_entries["FRONTEND_PYTHON"].get()
        APP_PATH = path_entries["APP_PATH"].get()
        CONFIG_PATH_UI = path_entries["CONFIG_PATH_UI"].get()
        config_win.destroy()

    tk.Button(config_win, text="💾 Save", command=save_paths, bg="#ff8800", fg="white", font=("Segoe UI", 11, "bold"), width=20).grid(row=5, column=1, pady=20)


# ========== GUI ==========
app = tk.Tk()
app.title("⚽ Football AI Pipeline")
app.geometry("700x360")
app.configure(bg="#1e1e1e")

# Title
title = tk.Label(app, text="⚙️ Football Video Tool", font=("Segoe UI", 18, "bold"), fg="#ffffff", bg="#1e1e1e")
title.pack(pady=20)

# Run Buttons
run_frame = tk.Frame(app, bg="#1e1e1e")
run_frame.pack(pady=5)
tk.Button(run_frame, text="🌐 Run Public", command=lambda: run_in_thread("public"), font=("Segoe UI", 11, "bold"), bg="#45cd3a", fg="white", width=38, height=2).pack(side="left", padx=10)

# Stop Button
stop_frame = tk.Frame(app, bg="#1e1e1e")
stop_frame.pack(pady=10)
tk.Button(stop_frame, text="🛑 Stop Server", command=stop_all, font=("Segoe UI", 11, "bold"), bg="#cc0000", fg="white", width=38, height=2).pack()

# Config Button
tk.Button(app, text="🛠️ Configure Paths", command=open_config_window, font=("Segoe UI", 11, "bold"), bg="#ff8800", fg="white", width=38, height=2).pack(pady=10)

# Periodically check for errors from threads
try:
    check_for_errors_in_gui(app, messagebox)
    app.mainloop()
except Exception as e:
    print("❌ Error in mainloop:", str(e))
