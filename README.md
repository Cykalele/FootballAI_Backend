# FootballAI – Tactical Video Analysis for Amateur Football

**FootballAI** is an AI-powered video analytics pipeline that enables amateur clubs, youth teams, and coaches to access professional-level tactical insights from standard camera recordings (drone, tripod, smartphone).  
The system processes raw match video through an end-to-end chain including player tracking, ball detection, tactical annotation, and metric computation — all without wearable sensors.

---

## 🚀 Features

- ⚽ **YOLOv12-based Object Detection** – Detects players, referees, and the ball.  
- 🎥 **Player Tracking with BoT-SORT** – Robust multi-object tracking with Re-ID and optical-flow fallback.  
- 🧠 **Ball Possession, Pass & Shot Analysis** – Detects passes, shots, and possession phases.  
- 🗺️ **2D Pitch Projection** – Transforms coordinates into a metric pitch view using keypoint-based homography.  
- 📊 **Live Metric Annotation** – Displays possession, shots, distance covered, Voronoi control zones, and more.  
- 🖼️ **Minimap Overlay** – Tactical radar view with live control zones.  
- 📑 **Metric Export** – Frame-wise and player-wise Excel summaries.  

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

FootballAI/
├── api/ # FastAPI backend for handling routes and logic
│ ├── routes/ # REST endpoints
│ │ ├── process_routes.py
│ │ ├── result_routes.py
│ │ ├── team_routes.py
│ │ └── upload_routes.py
│ ├── services/ # Backend orchestration & helpers
│ │ ├── pipeline_runner.py
│ │ ├── video_utils.py
│ │ ├── session_manager.py
│ │ └── start_server_pipeline.py
│
├── config/
│ └── settings.py # Central configuration for paths and parameters
│
├── core/
│ ├── annotators/ # Video overlay & annotation components
│ │ ├── ball_annotator.py
│ │ ├── heatmap_annotator.py
│ │ ├── metric_annotator.py
│ │ ├── minimap_annotator.py
│ │ ├── players_annotator.py
│ │ └── VideoAnnotator.py
│ │
│ ├── metrics/ # Metric computation modules
│ │ ├── ball_posession.py
│ │ ├── detect_goal.py
│ │ ├── metric_aggregator.py
│ │ ├── metrics_analyzer.py
│ │ ├── player_metrics.py
│ │ └── spatial_metrics.py
│ │
│ ├── team_assignment/
│ │ └── team_assignment.py # Automatic + manual team assignment
│ │
│ ├── tracking/ # Object detection & tracking
│ │ ├── ball_tracker.py
│ │ ├── object_detector.py
│ │ ├── player_tracker.py
│ │ └── VideoTracker.py
│ │
│ └── view_transformation/ # Keypoint-based pitch calibration
│ │ ├──── utils/
│ │ │ ├── averaged_View_transformer.py
│ │ │ ├── filter_objects.py
│ │ │ ├── pitch_utils.py
│ │ │ └── SmallFieldConfiguration.py
│ │ └── transformer.py
│
├── models/ # Pretrained models for inference
│ ├── cae_model.h5
│ ├── clip_market1501.pt
│ ├── MSMT17_clipreid.pth
│ ├── osnet_ain_x1_0_market1501.pth
│ ├── osnet_ain_x1_0_msmt17.pth
│ ├── osnet_x0_25_msmt17.pt
│ ├── Pose_Model.pt
│ └── Yolo12s_finetuned.pt
│
├── sessions/ # Output directory for processed sessions
│ └── <session_id>/ # Example: 20250808_095903_14f640f8
│   ├── annotated_game.mp4
│   ├── input.mp4
│   ├── tracking_log.json
│   ├── team_assignments.json
│   ├── ball_posession.json
│   ├── ball_events.json
│   ├── player_metrics.json
│   ├── spatial_metrics.json
│   ├── progress.json
│   ├── annotation_done.flag
│   └── heatmaps/
│
├── footballAI_env/ # Local Python virtual environment
│
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── start.bat # Launch backend & frontend
├── Aufbau.docx # Documentation (German)
└── cloudflared.exe # Tunnel binary for public access


