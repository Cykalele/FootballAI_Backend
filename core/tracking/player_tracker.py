import time
import numpy as np
import cv2
from boxmot import BotSort
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class PlayerTracker:
    """
    Tracks football players in video frames using BoT-SORT with ReID features.
    The tracker maintains last known positions and missing counters in order to support
    temporary gaps in detections. An optional optical-flow fallback remains available
    as commented code for scenarios with short-term occlusions.
    """

    def __init__(self, reid_model_path, enable_logging: bool = False) -> None:
        """
        Initializes the player tracker with BoT-SORT and a ReID backbone.

        Parameters
        ----------
        reid_model_path
            Path to the ReID weights to be used by BoT-SORT.
        enable_logging
            If set to True, additional diagnostic messages can be printed.

        Notes
        -----
        The device is selected as CUDA if available and otherwise as CPU.
        The configuration mirrors the original parameters to preserve behavior.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reid_model_path = Path(reid_model_path)

        self.tracker = BotSort(
            reid_weights=reid_model_path,
            device=device,
            half=False,
            track_buffer=200,
            with_reid=True,
            match_thresh=0.5,
            new_track_thresh=0.6,
            appearance_thresh=0.25,
            proximity_thresh=0.45,
            cmc_method="ecc",
            frame_rate=30,
            fuse_first_associate=True,
        )

        self.prev_gray = None
        self.last_positions: Dict[int, List[float]] = {}
        self.missing_counter: Dict[int, int] = defaultdict(int)
        # Stores optional ReID trajectories as (frame_idx, feature) tuples per track id.
        self.reid_trajectories: Dict[int, List[Any]] = defaultdict(list)

        # Maximum number of frames for which a missing track would be tolerated in fallback mode.
        self.max_missing = 5

        self.total_frames_tracked = 0
        self.enable_logging = enable_logging

    def track(self, detections, frame) -> List[Dict[str, Any]]:
        """
        Updates player tracks for a given frame and returns serialized track states.

        Parameters
        ----------
        detections
            Supervision detections for the current frame restricted to player class ids.
            The expected fields are xyxy, confidence, and class_id.
        frame
            The current video frame in BGR color space.

        Returns
        -------
        list of dict
            A list of player dictionaries with keys id, bbox, and class_id.
            The bbox is returned in xyxy format as floats.

        Notes
        -----
        The method does not modify the detection logic and adheres to the original
        serialization scheme used by the surrounding pipeline.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        players: List[Dict[str, Any]] = []
        tracked_ids = set()

        self.total_frames_tracked += 1

        # Prepare detections for BoT-SORT: [x1, y1, x2, y2, conf, cls]
        if len(detections.xyxy) > 0:
            det_array = np.column_stack(
                (detections.xyxy, detections.confidence, detections.class_id)
            )
        else:
            det_array = np.empty((0, 6))

        # Main tracking with ReID
        if det_array.shape[0] > 0:
            tracked = self.tracker.update(det_array, frame)
            for det in tracked:
                x1, y1, x2, y2, obj_id = det[:5]
                class_id = int(det[6])
                bbox = [x1, y1, x2, y2]

                players.append(
                    {
                        "id": int(obj_id),
                        "bbox": bbox,
                        "class_id": class_id,
                    }
                )

                tracked_ids.add(int(obj_id))
                self.last_positions[int(obj_id)] = bbox
                self.missing_counter[int(obj_id)] = 0

        self.prev_gray = gray.copy()
        return players
