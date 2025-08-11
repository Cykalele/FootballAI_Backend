# object_detector.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from config.settings import THREAD_WORKERS, SESSION_ROOT  # SESSION_ROOT kept for external consistency


class ObjectDetector:
    """
    This class wraps Ultralytics YOLO for object detection in football videos.
    It supports a tile-based inference strategy that improves the recall of small objects such as the ball.
    The detector can alternatively operate in a speed mode that performs a single full-frame inference per image.
    """

    def __init__(
        self,
        model_path: str,
        speed_tracking: bool,
        session_id: str,
        confidence_threshold: float,
        verbose: bool = True,
    ) -> None:
        """
        The constructor loads and fuses the YOLO model and prepares paths for session-specific diagnostics.
        The confidence threshold governs which raw detections are returned to the downstream tracking modules.

        Parameters
        ----------
        model_path
            This is the filesystem path to the YOLO weights file with extension .pt.
        speed_tracking
            This flag selects the full-frame mode when set to True and selects the tile-based mode otherwise.
        session_id
            This identifier names the current processing session and is used for organizing logs and artifacts.
        confidence_threshold
            This value defines the minimum confidence that a detection must exceed to be retained.
        verbose
            This flag controls the verbosity of diagnostic messages.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.speed_tracking = bool(speed_tracking)
        self.session_id = str(session_id)
        self.verbose = bool(verbose)

        self.model = YOLO(model_path)
        # Model fusion slightly speeds up inference and is safe for export-only usage.
        self.model.fuse()

        # Optional diagnostics for later inspection.
        self.rejected_log_path = Path(f"./sessions/{self.session_id}/rejected_bboxes.json")
        self.rejected_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------ Low-level inference ------------------------------

    def _detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """
        This method applies YOLO inference on a single image without spatial slicing.
        The input is expected in BGR channel order as commonly used by OpenCV.
        The method returns an empty Supervision Detections object when the model produces no boxes.

        Parameters
        ----------
        frame
            This array contains the image data of shape H by W by 3 in BGR order.

        Returns
        -------
        sv.Detections
            This object contains bounding boxes in xyxy format, confidence scores, and integer class identifiers.
        """
        result = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]

        if result.boxes is None:
            return sv.Detections.empty()

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        return sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

    def _tile_inference(self, frame: np.ndarray) -> sv.Detections:
        """
        This method performs tile-based inference to improve the detectability of small objects.
        The frame is partitioned into overlapping tiles with minimum dimensions to avoid degenerate patches.
        The method aggregates detections across all tiles while suppressing overlaps by non-maximum suppression.

        Parameters
        ----------
        frame
            This array contains the image to be processed.

        Returns
        -------
        sv.Detections
            This object contains the aggregated detections across all tiles.
        """
        MIN_TILE_WIDTH = 200
        MIN_TILE_HEIGHT = 150

        def callback(patch: np.ndarray) -> sv.Detections:
            try:
                h, w, _ = patch.shape
                if w < MIN_TILE_WIDTH or h < MIN_TILE_HEIGHT:
                    return sv.Detections.empty()
                return self._detect_objects(patch)
            except Exception as e:
                print(f"[Tile] Patch inference failed with error: {e}")
                return sv.Detections.empty()

        try:
            h, w, _ = frame.shape
            tile_w = max(w // 3, MIN_TILE_WIDTH)
            tile_h = max(h // 2, MIN_TILE_HEIGHT)

            slicer = sv.InferenceSlicer(
                callback=callback,
                slice_wh=(tile_w, tile_h),
                overlap_wh=(int(tile_w * 0.35), int(tile_h * 0.35)),
                overlap_ratio_wh=None,
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
                iou_threshold=0.25,
            )
            return slicer(frame)
        except Exception as e:
            print(f"[Tile] Slicing inference failed with error: {e}")
            return sv.Detections.empty()

    # --------------------------------- Public API ---------------------------------

    def infer_batch(self, frames: List[np.ndarray]) -> Dict[int, sv.Detections]:
        """
        This method applies object detection to a list of frames in parallel worker threads.
        It returns a mapping from the frame index within the batch to the corresponding detections.
        The method guarantees that each index appears in the output even when an exception occurs during inference.

        Parameters
        ----------
        frames
            This list contains the input frames in BGR order.

        Returns
        -------
        dict
            This dictionary maps the local batch index to a Supervision Detections object.
        """
        detections_dict: Dict[int, sv.Detections] = {}
        infer_fn = self._detect_objects if self.speed_tracking else self._tile_inference

        with ThreadPoolExecutor(max_workers=int(THREAD_WORKERS)) as executor:
            futures = {i: executor.submit(infer_fn, frame) for i, frame in enumerate(frames)}

            for i, future in futures.items():
                try:
                    detections_dict[i] = future.result(timeout=20)
                except Exception as e:
                    print(f"[Detect] Inference failed on batch index {i} with error: {e}")
                    detections_dict[i] = sv.Detections.empty()

        return detections_dict
