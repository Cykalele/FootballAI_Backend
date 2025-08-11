# FootballAI – Tactical Video Analysis for Amateur Football

**FootballAI** is an AI-powered video analytics system designed to provide amateur clubs, youth academies, and coaches with professional-grade tactical insights.  
The system processes standard match footage from a drone through a complete end-to-end pipeline, including player and ball detection, tactical annotation, and spatio-temporal metric computation — entirely without wearable sensors.

## ⚠️ Important Note on System Structure

The backend **must** be used together with the dedicated **FootballAI Frontend**.  
Both must be placed within a shared parent directory.

---

## 🚀 Key Features

- **YOLOv12-based Object Detection** – Accurate detection of players, referees, and the ball.  
- **Player Tracking with BoT-SORT** – Robust multi-object tracking with re-identification and optical-flow fallback.  
- **Ball Possession, Pass, and Shot Analysis** – Automatic identification of key match events.  
- **2D Pitch Projection** – Transforms coordinates into a metric pitch view using keypoint-based homography.  
- **Live Metric Annotation** – Displays possession, shots, distances covered, Voronoi control zones, and more.  
- **Minimap Overlay** – Tactical radar view with live control zones.  
- **Metric Export** – Frame-wise and player-wise Excel reports.

---

## 📦 Installation

1. **Clone or Download the Repository**  
   Either download the repository from GitHub or clone it:
   ```powershell
   git clone https://github.com/Cykalele/FootballAI_Backend.git

2. **Place Cloudflare Tunnel Binary**
   Place the file cloudflared.exe in the project’s root directory.
   This enables public access to the Streamlit frontend without manual configuration.

3. **Python Environment Initialization**
   All required Python dependencies are installed and a dedicated virtual environment is automatically initialized when running start.bat.
   No manual activation of the environment is required.

4. **Git LFS for Model Files**
   Ensure Git LFS is installed and activated to correctly download large model files (YOLO, Re-ID, pose models):
      git lfs install
      git lfs pull



---

## ▶️ Usage

1. **Run the system**  
   Double-click `start.bat`.  
   This opens the console window and starts the backend and frontend.  

2. **Public access**  
   In the console menu, choose **Run Public** to publish the frontend to:  

3. **Stopping the system**  
Either close the console window or choose **Stop Server**.  

4. **Configuring paths**  
Use **Configure Paths** to set:
- Backend Python path
- Frontend Python path
- `app.py` path for the frontend
- `config.py` path for the frontend  

---

## 📜 Acknowledgements

Parts of this repository include adapted code from  
[Roboflow Sports](https://github.com/roboflow/sports), licensed under the MIT License.  
Usage within classes and modules is marked accordingly in the source code.

---

## 📂 Project Structure

```plaintext
FootballAI_Backend/
├── api/                          # FastAPI backend for handling routes and logic
│   ├── routes/                   # REST endpoints
│   │   ├── process_routes.py
│   │   ├── result_routes.py
│   │   ├── team_routes.py
│   │   └── upload_routes.py
│   │
│   └── services/                 # Backend orchestration & helpers
│       ├── pipeline_runner.py
│       ├── video_utils.py
│       ├── session_manager.py
│       └── start_server_pipeline.py
│   
├── config/
│   └── settings.py               # Central configuration for paths and parameters
│   
├── core/
│   ├── annotators/               # Video overlay & annotation components
│   │   ├── ball_annotator.py
│   │   ├── heatmap_annotator.py
│   │   ├── metric_annotator.py
│   │   ├── minimap_annotator.py
│   │   ├── players_annotator.py
│   │   └── VideoAnnotator.py
│   │
│   ├── metrics/                  # Metric computation modules
│   │   ├── ball_posession.py
│   │   ├── detect_goal.py
│   │   ├── metric_aggregator.py
│   │   ├── metrics_analyzer.py
│   │   ├── player_metrics.py
│   │   └── spatial_metrics.py
│   │
│   ├── team_assignment/
│   │   └── team_assignment.py    # Automatic + manual team assignment
│   │
│   ├── tracking/                 # Object detection & tracking
│   │   ├── ball_tracker.py
│   │   ├── object_detector.py
│   │   ├── player_tracker.py
│   │   └── VideoTracker.py
│   │
│   └── view_transformation/      # Keypoint-based pitch calibration
│       ├── utils/
│       │   ├── averaged_View_transformer.py
│       │   ├── filter_objects.py
│       │   ├── pitch_utils.py
│       │   └── SmallFieldConfiguration.py
│       └── transformer.py
│
├── models/                       # Pretrained models for inference (Git LFS)
│   ├── cae_model.h5              # Model used for Team Assignment
│   ├── clip_market1501.pt        # Model used for Re-Identification during Tracking
│   ├── Pose_Model.pt             # Model used for Keypoint Detection in View Transformation
│   └── Yolo12s_finetuned.pt      # Model used for Player, Ball, Referee Detection in frame
│
├── sessions/                     # Output directory for processed sessions
│   └── .gitkeep                  # keep folder in repo; contents are generated
│   
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── start.bat                     # Launch backend & frontend
└── cloudflared.exe               # Tunnel binary for public access
