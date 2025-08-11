import os
import time
import cv2
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from collections import deque

from config.settings import POSE_MODEL_PATH
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration
from core.view_transformation.utils.averaged_View_transformer import AveragedViewTransformer
from core.view_transformation.utils.filter_objects import filter, filter_ball_coordinates


def draw_keypoints_and_edges(frame: np.ndarray, keypoints: np.ndarray, edges, labels=None) -> np.ndarray:
    """
    Draws detected keypoints and their connecting edges on a given frame.

    Parameters
    ----------
    frame : np.ndarray
        Original image frame.
    keypoints : np.ndarray
        Array of 2D keypoint coordinates (x, y).
    edges : list of tuple
        List of index pairs indicating which keypoints should be connected.
    labels : list of str, optional
        Labels for each keypoint, drawn next to the point if provided.

    Returns
    -------
    np.ndarray
        Frame with drawn keypoints and edges.
    """
    vis = frame.copy()
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), -1)
        if labels and i < len(labels):
            cv2.putText(vis, labels[i], (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw connecting edges
    for start_idx, end_idx in edges:
        i1, i2 = start_idx - 1, end_idx - 1
        if i1 >= len(keypoints) or i2 >= len(keypoints):
            continue
        pt1, pt2 = keypoints[i1], keypoints[i2]
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        cv2.line(vis, (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])), (0, 0, 255), 2)

    return vis


class ViewTransformer:
    """
    Estimates and applies a 2D homography transformation from image space
    to metric pitch coordinates using detected field keypoints.

    The transformation is stabilized by maintaining a buffer of recent
    homography matrices and computing a weighted average.
    """

    def __init__(self, window_size: int = 7):
        # Load YOLO pose estimation model for pitch keypoints
        self.model = YOLO(POSE_MODEL_PATH)
        self.pitch_config = SmallFieldConfiguration()

        # Buffer for temporal averaging of homographies (sliding window)
        self.window_size = max(3, int(window_size))
        self.H_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.H_avg = None

        # Cache of the most recently used transformer
        self.last_transformer = None
        self.last_update_frame = -1
        self.progress_path = None  # Path to store progress information

    def _weighted_average_H(self) -> np.ndarray:
        """
        Computes a linearly weighted average of buffered homography matrices.

        More recent matrices receive higher weights (1..N). The result is
        normalized such that H[2, 2] = 1.

        Returns
        -------
        np.ndarray
            Averaged homography matrix.
        """
        n = len(self.H_buffer)
        assert n > 0
        weights = np.arange(1, n + 1, dtype=np.float32)
        weights /= weights.sum()
        H_sum = np.zeros((3, 3), dtype=np.float32)
        for w, H in zip(weights, self.H_buffer):
            H_sum += w * H
        H_sum /= H_sum[2, 2]
        return H_sum

    def get_pitch_transformer(self, frame, frame_idx):
        """
        Estimates the homography from the current frame to the pitch reference.

        Steps:
        1. Detect pitch keypoints using YOLO pose model.
        2. Filter keypoints by confidence.
        3. Match detected image points to known pitch coordinates.
        4. Compute homography and update temporal average.

        Parameters
        ----------
        frame : np.ndarray
            Current video frame.
        frame_idx : int
            Frame index in the video.

        Returns
        -------
        AveragedViewTransformer or None
            Transformer that applies the averaged homography.
        """
        result = self.model(frame, verbose=False, conf=0.2)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Skip if no keypoints are detected
        if keypoints is None or not hasattr(keypoints, "confidence"):
            print(f"⚠️ Frame {frame_idx}: No keypoints – skip.")
            return self.last_transformer

        confidence = keypoints.confidence[0]
        xy = keypoints.xy[0]
        corrected_xy = xy
        confidence_filter = confidence > 0.5

        # Require at least 4 confident keypoints to compute homography
        if confidence_filter.sum() < 4:
            print(f"⚠️ Frame {frame_idx}: Too few keypoints.")
            return self.last_transformer

        # Map image keypoints to pitch vertices
        frame_pts = corrected_xy[confidence_filter]
        pitch_pts = np.array(self.pitch_config.vertices)[confidence_filter]

        H_new, _ = cv2.findHomography(frame_pts.astype(np.float32),
                                      pitch_pts.astype(np.float32))
        if H_new is None:
            return self.last_transformer

        H_new = H_new.astype(np.float32)
        H_new /= H_new[2, 2]

        # Update homography buffer and compute weighted average
        self.H_buffer.append(H_new.copy())
        self.H_avg = self._weighted_average_H()

        self.last_transformer = AveragedViewTransformer(homography_matrix=self.H_avg.copy())
        self.last_update_frame = frame_idx

        return self.last_transformer

    def run(self, video_path: str, tracking_log_path: str, verbose: bool = True):
        """
        Processes the video to compute and apply 2D pitch transformations
        for players and ball in each frame.

        Parameters
        ----------
        video_path : str
            Path to the source video.
        tracking_log_path : str
            Path to the JSON file containing detection tracking data.
        verbose : bool
            If True, saves debug images with drawn keypoints.
        """
        with open(tracking_log_path, "r") as f:
            tracking_data = json.load(f)

        # Prepare progress tracking
        session_path = os.path.dirname(tracking_log_path)
        progress_dir = os.path.join(session_path, "progress")
        os.makedirs(progress_dir, exist_ok=True)
        self.progress_path = os.path.join(progress_dir, "transformer_progress.json")

        if verbose:
            keypoint_dir = os.path.join(session_path, "keypoint_frames")
            os.makedirs(keypoint_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames = np.linspace(0, total - 1, 50, dtype=int) if verbose else []

        idx = 0
        self._save_progress(0, total)

        print("🧭 Running 2D View Transformation...")

        with tqdm(total=total, desc="ViewTransform", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_log = tracking_data.get(str(idx), {})
                players = frame_log.get("players", [])
                ball = frame_log.get("ball", [])

                # Only compute transformation if objects are present
                if players or ball:
                    transformer = self.get_pitch_transformer(frame, idx)
                    if transformer:
                        # === Player processing ===
                        player_boxes = [p["bbox"] for p in players if "bbox" in p and p["bbox"] is not None]
                        player_ids   = [p["id"]   for p in players if "bbox" in p and p["bbox"] is not None]

                        # Build detections object for player positions
                        if len(player_boxes) > 0:
                            dets_players = sv.Detections(
                                xyxy=np.array(player_boxes, dtype=np.float32).reshape(-1, 4)
                            )
                        else:
                            dets_players = sv.Detections.empty()

                        player_positions = dets_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

                        # If no valid positions or IDs exist, write empty player list
                        if player_positions.shape[0] == 0 or len(player_ids) == 0:
                            if str(idx) not in tracking_data:
                                tracking_data[str(idx)] = {}
                            tracking_data[str(idx)]["players"] = []
                        else:
                            pitch_coords = transformer.transform_points(player_positions, player_ids)

                            # Ensure dictionary format
                            if isinstance(pitch_coords, np.ndarray):
                                pitch_coords = {pid: pt for pid, pt in zip(player_ids, pitch_coords)}

                            pitch_coords = filter(pitch_coords)

                            valid_ids = set(pitch_coords.keys())
                            id_to_player = {p["id"]: p for p in players if p.get("id") in valid_ids}

                            for pid in id_to_player:
                                id_to_player[pid]["position_2d"] = pitch_coords[pid].tolist()

                            if str(idx) not in tracking_data:
                                tracking_data[str(idx)] = {}
                            tracking_data[str(idx)]["players"] = list(id_to_player.values())

                        # === Ball processing ===
                        ball_bboxes = [b["bbox"] for b in ball if "bbox" in b]

                        if len(ball_bboxes) > 0:
                            dets_ball = sv.Detections(
                                xyxy=np.array(ball_bboxes, dtype=np.float32).reshape(-1, 4)
                            )
                        else:
                            dets_ball = sv.Detections.empty()

                        ball_coords = dets_ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        pitch_ball = transformer.transform_points(ball_coords)
                        pitch_ball = filter_ball_coordinates(pitch_ball)

                        if "ball" not in tracking_data[str(idx)] or not tracking_data[str(idx)]["ball"]:
                            tracking_data[str(idx)]["ball"] = []

                        # Store ball coordinates in tracking log
                        if isinstance(pitch_ball, np.ndarray) and pitch_ball.shape[0] > 0:
                            tracking_data[str(idx)]["ball"] = [{
                                "id": 0,
                                "bbox": tracking_data[str(idx)]["ball"][0]["bbox"] if tracking_data[str(idx)]["ball"] else [None, None, None, None],
                                "class_id": 0,
                                "position_2d": pitch_ball[0].tolist()
                            }]
                        else:
                            tracking_data[str(idx)]["ball"] = []

                idx += 1
                self._save_progress(idx, total)
                pbar.update(1)

        cap.release()

        # Save updated tracking log with 2D positions
        with open(tracking_log_path, "w") as f:
            json.dump(tracking_data, f, indent=4)

        self._save_progress(total, total)

        # Create completion flag file
        done_flag_path = os.path.join(os.path.dirname(tracking_log_path),
                                      "progress", "view_transformer_done.flag")
        with open(done_flag_path, "w") as f:
            f.write("done")

        print("✅ View transformation completed. 2D coords stored in tracking log.")

    def _save_progress(self, current, total):
        """
        Writes progress information (current frame, total frames) to disk.

        Parameters
        ----------
        current : int
            Current frame index processed.
        total : int
            Total number of frames in the video.
        """
        if not self.progress_path:
            return
        with open(self.progress_path, "w") as f:
            json.dump({"current": current, "total": total}, f)
