import subprocess
import os
os.add_dll_directory(r"C:\Users\hause\OneDrive\01_Studium\02_Master\05_Masterarbeit\02_FootballAI\footballAI_env\Lib\site-packages\cv2")

import cv2
import json
import numpy as np
from tqdm import tqdm
import supervision as sv
import threading
import queue
from core.annotators.players_annotator import PlayerAnnotator
from core.annotators.ball_annotator import BallAnnotator
from core.annotators.minimap_annotator import MinimapAnnotator
from core.metrics.metrics_analyzer import MetricsAnalyzer


class VideoAnnotator:
    """
    Orchestrates the annotation of football videos with player, ball, and minimap overlays.
    The class processes a precomputed tracking log, applies visual overlays using dedicated annotators,
    integrates metric updates, and writes the annotated video to disk.
    """

    def __init__(self, tracking_log_path, assignment_path, output_video_path, metric_path, team_config, session_id):
        """
        Initializes the video annotation pipeline and instantiates annotators.

        Parameters
        ----------
        tracking_log_path : str
            Path to the JSON file containing per-frame tracking data.
        assignment_path : str
            Path to the JSON file containing team assignments.
        output_video_path : str
            Path to the output annotated video file.
        metric_path : str
            Path to save computed metrics.
        team_config : dict
            Dictionary describing team-specific visual parameters (e.g., colors).
        session_id : str
            Identifier for the current processing session, used for progress tracking.
        """
        self.tracking_log_path = tracking_log_path
        self.assignment_path = assignment_path
        self.output_video_path = output_video_path
        self.metric_path = metric_path
        self.team_config = team_config
        self.session_id = session_id

        self.player_annotator = PlayerAnnotator(team_config)
        self.ball_annotator = BallAnnotator(radius=7, buffer_size=10, thickness=2)
        self.shared_minimap_annotator = MinimapAnnotator(team_config)
        self.metrics_analyzer = MetricsAnalyzer(metric_path, assignment_path, team_config, session_id)

        progress_dir = f"./sessions/{session_id}/progress"
        os.makedirs(progress_dir, exist_ok=True)
        self.progress_path = os.path.join(progress_dir, "annotator_progress.json")

    def annotate_frame(self, frame_idx, frame, frame_log, team_assignments, minimap_annotator):
        """
        Annotates a single frame with player boxes, ball markers, and a minimap overlay.

        Parameters
        ----------
        frame_idx : int
            Index of the frame in the video sequence.
        frame : np.ndarray
            The current video frame in BGR format.
        frame_log : dict
            Tracking log entry for the current frame.
        team_assignments : dict
            Mapping from player ID to assigned team ID.
        minimap_annotator : MinimapAnnotator
            Shared instance of the minimap annotator.

        Returns
        -------
        tuple
            (annotated_frame, pitch_ball_coords, pitch_player_coords_dict, player_ids_list, control_mask)
        """
        players = frame_log.get("players", [])
        ball = frame_log.get("ball", [])

        if not players and not ball:
            return frame, None, None, [], None

        pitch_player_coords_dict = {
            p["id"]: np.array(p["position_2d"], dtype=np.float32)
            for p in players
        }

        pitch_ball_coords = (
            np.array(ball[0]["position_2d"], dtype=np.float32)
            if ball and isinstance(ball[0], dict) and "position_2d" in ball[0]
            else None
        )

        annotated_frame, player_coords_team_dict = self.player_annotator.annotate(
            frame, players, pitch_player_coords_dict, team_assignments
        )

        if ball and "bbox" in ball[0]:
            ball_detections_sv = sv.Detections(
                xyxy=np.array([ball[0]["bbox"]], dtype=np.float32)
            )
            annotated_frame = self.ball_annotator.annotate(annotated_frame, ball_detections_sv)
        else:
            ball_detections_sv = sv.Detections.empty()

        if pitch_ball_coords is not None:
            pitch_ball_coords = pitch_ball_coords.reshape(1, 2)

        minimap, control_mask = minimap_annotator.annotate_pitch_diagramm(
            frame_idx, pitch_ball_coords, player_coords_team_dict
        )
        annotated_frame = minimap_annotator.overlay_minimap(annotated_frame, minimap)

        player_ids_list = [int(p["id"]) for p in players]

        return annotated_frame, pitch_ball_coords, pitch_player_coords_dict, player_ids_list, control_mask

    def annotate_video_from_path(self, video_path: str):
        """
        Runs the full annotation pipeline on the given video file.

        Parameters
        ----------
        video_path : str
            Path to the input video to annotate.

        Notes
        -----
        The function uses a producer-consumer threading model to parallelize
        frame annotation and metric updates while writing frames in sequence.
        """
        with open(self.progress_path, "w") as f:
            json.dump({"current": 0, "total": 500}, f)

        with open(self.tracking_log_path, "r") as f:
            tracking_data_raw = json.load(f)
            tracking_data = {int(k): v for k, v in tracking_data_raw.items()}

        try:
            with open(self.assignment_path, "r") as f:
                team_assignments_raw = json.load(f)
                team_assignments = {
                    int(pid): pdata["team"]
                    for pid, pdata in team_assignments_raw["players"].items()
                    if "team" in pdata and not pdata.get("removed", False)
                }
        except FileNotFoundError:
            print(f"[VideoAnnotator] {self.assignment_path} not found – team overlays disabled.")
            team_assignments = {}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError("VideoWriter could not be initialized.")

        print(f"[VideoAnnotator] Annotating video: {video_path} → {self.output_video_path}")

        frame_queue = queue.Queue(maxsize=200)
        result_queue = queue.Queue(maxsize=200)
        NUM_WORKERS = min(2, os.cpu_count())

        def producer():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_log = tracking_data.get(idx, {})
                frame_queue.put((idx, frame, frame_log))
                idx += 1
            for _ in range(NUM_WORKERS):
                frame_queue.put(None)

        def worker():
            while True:
                item = frame_queue.get()
                if item is None:
                    result_queue.put(None)
                    break
                idx, frame, frame_log = item
                result = self.annotate_frame(idx, frame, frame_log, team_assignments, self.shared_minimap_annotator)
                result_queue.put((idx, frame_log, *result))

        def consumer():
            done_workers = 0
            last_reported_percent = -1
            buffer = {}
            next_expected_idx = 0

            with tqdm(total=total_frames, desc="Annotating Video", unit="frame") as pbar:
                while done_workers < NUM_WORKERS:
                    item = result_queue.get()
                    if item is None:
                        done_workers += 1
                        continue

                    idx, frame_log, annotated_frame, pitch_ball_coords, pitch_player_coords_dict, player_ids_list, control_mask = item

                    buffer[idx] = (
                        annotated_frame,
                        pitch_ball_coords,
                        pitch_player_coords_dict,
                        player_ids_list,
                        control_mask,
                        frame_log,
                    )

                    while next_expected_idx in buffer:
                        (
                            annotated_frame,
                            pitch_ball_coords,
                            pitch_player_coords_dict,
                            player_ids_list,
                            control_mask,
                            frame_log,
                        ) = buffer.pop(next_expected_idx)

                        if pitch_ball_coords is not None:
                            self.metrics_analyzer.update(
                                next_expected_idx,
                                pitch_ball_coords,
                                pitch_player_coords_dict,
                                player_ids_list,
                                control_mask,
                                frame_log,
                                team_assignments,
                            )
                            annotated_frame = self.metrics_analyzer.annotate_overlay(annotated_frame, next_expected_idx)

                        out.write(annotated_frame)

                        percent = int((next_expected_idx / total_frames) * 100)
                        if percent != last_reported_percent:
                            update_annotation_progress(self.progress_path, current=next_expected_idx, total=total_frames)
                            last_reported_percent = percent

                        pbar.update(1)
                        next_expected_idx += 1

            if last_reported_percent < 100:
                update_annotation_progress(self.progress_path, current=total_frames, total=total_frames)

        producer_thread = threading.Thread(target=producer)
        worker_threads = [threading.Thread(target=worker) for _ in range(NUM_WORKERS)]
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        for t in worker_threads:
            t.start()
        consumer_thread.start()

        producer_thread.join()
        for t in worker_threads:
            t.join()
        consumer_thread.join()

        cap.release()
        out.release()

        try:
            self.metrics_analyzer.save_all()
            print(f"[VideoAnnotator] Metrics saved to {self.metric_path}")
        except Exception as e:
            print(f"[VideoAnnotator] Failed to save metrics: {e}")

        try:
            done_flag_path = os.path.join("./sessions", self.session_id, "progress", "annotation_done.flag")
            with open(done_flag_path, "w") as f:
                f.write("done")
            print(f"[VideoAnnotator] Annotation completed: {done_flag_path} created.")
        except Exception as e:
            print(f"[VideoAnnotator] Failed to set completion flag: {e}")

        print(f"[VideoAnnotator] Annotated video saved to {self.output_video_path}")


def update_annotation_progress(progress_path, current, total):
    """
    Updates the annotation progress JSON file with the current frame and total frame count.

    Parameters
    ----------
    progress_path : str
        Path to the progress JSON file.
    current : int
        Current frame index processed.
    total : int
        Total number of frames in the video.
    """
    try:
        with open(progress_path, "w") as f:
            json.dump({"current": current, "total": total}, f)
    except Exception as e:
        print(f"[VideoAnnotator] Could not save progress (frame {current}): {e}")
