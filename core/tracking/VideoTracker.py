import os
import cv2
import json
import orjson
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import supervision as sv
import torch  # kept due to external dependencies that may rely on CUDA initialization

from config.settings import MODEL_PATH, OBJECT_DETECTION_THRESHOLD
from core.tracking.object_detector import ObjectDetector
from core.tracking.player_tracker import PlayerTracker
from core.tracking.ball_tracker import BallTracker


class VideoTracker:
    """
    Orchestrates the video-based tracking pipeline for football analytics.

    The class consumes an MP4 video and executes batched object detection,
    player and ball tracking, and frame-wise serialization of tracking results
    into a compact JSON log. It is designed for high-throughput processing with
    bounded memory by decoupling detection and tracking through a producer–
    consumer queue.

    Parameters
    ----------
    video_path : str | Path
        Path to the input video file in MP4 format.
    reid_model_path : str | Path
        Path to the ReID model used by the PlayerTracker.
    tracking_output_path : str | Path
        Path to the output JSON file that will store frame-wise tracking results.
        The file is written as a single JSON object with frame indices as keys.
    session_id : str
        Identifier of the processing session used to persist progress artifacts.
    speed_tracking : bool, optional
        Enables a lighter configuration of the object detector for faster
        inference at the expense of accuracy. Defaults to False.
    force_track : bool, optional
        Forces re-tracking even if an output file already exists. Defaults to False.

    Notes
    -----
    The JSON structure produced by this orchestrator follows:
        {
          "<frame_idx>": {
            "players": [{"id": int, "bbox": [x1, y1, x2, y2], "class_id": 1}, ...],
            "ball":    [{"id": 0,   "bbox": [x1, y1, x2, y2], "class_id": 0}, ...]
          },
          ...
        }

    The queue connects one detection producer with one tracking consumer.
    A sentinel value of None signals the end of the stream.
    """

    def __init__(
        self,
        video_path,
        reid_model_path,
        tracking_output_path,
        session_id: str,
        speed_tracking: bool = False,
        force_track: bool = False,
    ) -> None:
        self.video_path = str(video_path)
        self.tracking_output_path = str(tracking_output_path)
        self.force_track = force_track
        self.speed_tracking = speed_tracking

        self.detector = ObjectDetector(
            MODEL_PATH,
            speed_tracking,
            session_id,
            confidence_threshold=OBJECT_DETECTION_THRESHOLD,
        )
        self.player_tracker = PlayerTracker(reid_model_path, enable_logging=True)
        self.ball_tracker = BallTracker()

        self.progress_dir = f"./sessions/{session_id}/progress"
        os.makedirs(self.progress_dir, exist_ok=True)
        self.progress_path = os.path.join(self.progress_dir, "tracking_progress.json")
        os.makedirs(os.path.dirname(self.tracking_output_path), exist_ok=True)

        # The queue buffers batches from detection for tracking.
        self.frame_queue: "Queue[Optional[Tuple[int, np.ndarray, sv.Detections]]]" = Queue(maxsize=256)
        self.batch_size: int = 128

    # --------------------------- Internal pipeline stages ---------------------------

    def run_detection(self, cap: cv2.VideoCapture, total_frames: int) -> None:
        """
        Performs batched object detection and enqueues frames with their detections.

        This method reads frames from an open cv2.VideoCapture handle in fixed-size
        batches, runs the detector once per batch, and pushes tuples of
        (absolute_frame_index, frame, detections) to the shared queue.

        Parameters
        ----------
        cap : cv2.VideoCapture
            An opened video capture handle for the input video.
        total_frames : int
            Total number of frames in the input video. Used only for progress coupling.

        Side Effects
        ------------
        Enqueues items into `self.frame_queue`. After the last batch, a sentinel
        value of None is enqueued to signal termination to the consumer.
        """
        frame_idx = 0
        while True:
            batch: List[np.ndarray] = []

            # Read up to batch_size frames or until the video ends.
            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append(frame)

            if not batch:
                break

            # Run a single batched inference call.
            batch_detections: Dict[int, sv.Detections] = self.detector.infer_batch(batch)

            # Enqueue frames and their detections with absolute indices.
            for i, frame in enumerate(batch):
                abs_idx = frame_idx + i
                detections = batch_detections.get(i, sv.Detections.empty())
                self.frame_queue.put((abs_idx, frame, detections))

            frame_idx += len(batch)

        # Signal end of stream.
        self.frame_queue.put(None)

    def run_tracking(self, log_file, total_frames: int) -> None:
        """
        Consumes frames with detections, performs tracking, and streams results to JSON.

        Parameters
        ----------
        log_file : io.BufferedWriter
            An open binary file handle to which JSON fragments are written.
            The caller is responsible for writing the opening and closing braces.
        total_frames : int
            Total number of frames for progress reporting.

        Notes
        -----
        Player tracks are produced by the PlayerTracker and serialized with
        stable integer identifiers. Ball positions are updated by the BallTracker.
        All values are cast to primitive types for compact JSON encoding with orjson.
        """
        pbar = tqdm(total=total_frames, desc="Tracking video", unit="frame")
        first_entry = True

        while True:
            item = self.frame_queue.get()
            if item is None:
                break

            abs_idx, frame, detections = item

            try:
                # Ball tracking
                ball_detections = detections[detections.class_id == 0]
                ball_tracked = self.ball_tracker.update(frame, ball_detections)

                # Player tracking
                player_detections = detections[detections.class_id == 1]
                tracked_players = self.player_tracker.track(player_detections, frame)

                # Serialize players
                players_out = [
                    {
                        "id": int(p["id"]),
                        "bbox": [float(x) for x in p["bbox"]],
                        "class_id": int(p.get("class_id", 1)),
                    }
                    for p in tracked_players
                ]

                # Serialize ball detections after tracker update
                ball_out: List[Dict] = []
                ids = ball_tracked.class_id if ball_tracked.class_id is not None else [0] * len(ball_tracked)
                for xyxy, class_id in zip(ball_tracked.xyxy, ids):
                    x1, y1, x2, y2 = xyxy
                    ball_out.append(
                        {
                            "id": 0,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "class_id": int(class_id),
                        }
                    )

                # Stream a single frame entry without loading all frames into memory
                frame_entry = orjson.dumps(
                    {
                        str(abs_idx): {
                            "players": players_out,
                            "ball": ball_out,
                        }
                    }
                )[1:-1]  # remove surrounding braces to build a single JSON object incrementally

                if not first_entry:
                    log_file.write(b",\n")
                log_file.write(frame_entry)
                first_entry = False

                self._save_progress(abs_idx + 1, total_frames)
                pbar.update(1)

            except Exception as e:
                # The pipeline must remain robust under per-frame failures.
                print(f"[ERROR] Exception during tracking at frame {abs_idx}: {e}")
                continue

        pbar.close()

    # ------------------------------- Public entry point -------------------------------

    def track_video(self) -> None:
        """
        Executes the complete tracking pipeline on the configured video.

        The method opens the video, launches the detection producer and tracking
        consumer as dedicated threads, and writes a frame-indexed JSON file to
        `self.tracking_output_path`. If the output exists and `force_track` is False,
        the method returns immediately.

        Raises
        ------
        IOError
            If the input video cannot be opened.
        """
        if os.path.exists(self.tracking_output_path) and not self.force_track:
            print(f"Tracking log already exists at {self.tracking_output_path}. Skipping re-tracking.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video at path {self.video_path}.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Starting tracking for video at {self.video_path}.")

        # Stream JSON output to disk
        log_file = open(self.tracking_output_path, "wb")
        log_file.write(b"{\n")

        try:
            producer = Thread(target=self.run_detection, args=(cap, total_frames), daemon=True)
            consumer = Thread(target=self.run_tracking, args=(log_file, total_frames), daemon=True)

            producer.start()
            consumer.start()

            producer.join()
            consumer.join()
        finally:
            # Close JSON object and release resources
            log_file.write(b"\n}")
            log_file.close()
            cap.release()

        # Persist a simple completion flag for external progress monitors
        flag_path = os.path.join(self.progress_dir, "tracker_done.flag")
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("done")

        print(f"Tracking completed. Results written to {self.tracking_output_path}.")

    # --------------------------------- Auxiliaries ---------------------------------

    def _save_progress(self, current: int, total: int) -> None:
        """
        Persists the current progress to a JSON file for front-end polling.

        Parameters
        ----------
        current : int
            The number of frames already processed.
        total : int
            The total number of frames in the video.
        """
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump({"current": int(current), "total": int(total)}, f)
