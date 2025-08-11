import json
import os
import cv2
import numpy as np
import requests
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model, Model
from config.settings import CAE_PATH, API_BASE_PATH
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from core.view_transformation.utils.SmallFieldConfiguration import SmallFieldConfiguration
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def weighted_average(score_list):
    """
    Computes a weighted arithmetic mean from a list of (score, weight) pairs.
    The function returns None when the input list is empty or when the sum of weights is not positive.
    The calculation remains numerically stable for small lists and typical floating-point weights.
    """
    if not score_list:
        return None
    total_weight = sum(w for _, w in score_list)
    if total_weight <= 0:
        return None
    return sum(score * weight for score, weight in score_list) / total_weight


class TeamAssignment:
    """
    Assigns team labels to player tracks using appearance features from a convolutional autoencoder.
    The class extracts features from cropped player bounding boxes, reduces dimensionality with principal component analysis,
    and performs two-cluster KMeans to separate team identities. The procedure operates window-wise along the video timeline
    in order to utilize first occurrences of players and to limit memory usage. The method persists assignments, estimated team
    attack directions, and diagnostic plots, and it optionally notifies a backend for manual correction workflows.
    """

    def __init__(
        self,
        tracking_log_path,
        video_path,
        assignment_output_path,
        team_config: dict,
        session_id: str,
        run_manual_assignment: bool,
    ):
        """
        Creates the assignment component and loads all resources required for feature extraction and clustering.
        The constructor loads the tracking log into memory, opens the convolutional autoencoder and builds an encoder submodel.
        The field geometry is read from the SmallFieldConfiguration in order to mask implausible positions.

        The parameters are interpreted as follows. The tracking_log_path points to a JSON file with per-frame player states.
        The video_path refers to the source MP4. The assignment_output_path indicates the JSON file that will store player to team mappings
        and direction estimates. The team_config contains optional metadata for the frontend. The session_id namespaces artifacts on disk.
        The run_manual_assignment flag selects whether a backend notification should trigger a manual review phase.
        """
        with open(tracking_log_path, "r") as f:
            self.tracking_log = json.load(f)

        self.video_path = video_path
        self.assignment_output = assignment_output_path
        self.team_config = team_config
        self.session_id = session_id
        self.api_base = os.getenv("API_BASE", "http://localhost:8000")
        self.run_manual_assignment = run_manual_assignment

        # Will be filled with entries of the form {player_id: {"team": int, "bbox": [x1,y1,x2,y2], "removed": bool}}
        self.team_assignments: Dict[int, Dict] = {}
        # Default direction configuration applies when no evidence suggests otherwise.
        self.direction_config = {"1": "left_to_right", "2": "right_to_left"}

        # Window-wise quality scores are stored with weights for later aggregation.
        self.silhouette_scores_weighted: List[Tuple[float, int]] = []
        self.dbi_scores_weighted: List[Tuple[float, int]] = []

        # Field geometry in centimeters used for positional plausibility checks.
        self.field_length_cm = SmallFieldConfiguration.length
        self.field_width_cm = SmallFieldConfiguration.width

        # Load CAE and expose the named encoder layer for feature extraction.
        model = load_model(CAE_PATH)
        self.encoder: Model = Model(inputs=model.input, outputs=model.get_layer("conv2d_15").output)

    def api_url(self, endpoint: str) -> str:
        """
        Builds a fully qualified URL to the backend endpoint. The method respects the base path configured in API_BASE_PATH
        and concatenates it with the session specific subresource. No network request is performed here.
        """
        return f"{self.api_base.rstrip('/')}{API_BASE_PATH.rstrip('/')}/{endpoint.lstrip('/')}"

    def assign_teams(self) -> None:
        """
        Executes the complete team assignment workflow on the source video and tracking log.
        The procedure iterates over fixed windows of frames and collects first appearances of players within each window.
        For each first occurrence the method crops the corresponding player patch, extracts a CAE feature, and gathers features across the window.
        The method performs dimensionality reduction with principal component analysis and KMeans clustering with two clusters.
        Cluster labels are mapped to team identifiers by frequency, and results are recorded in the assignment dictionary.
        After all windows are processed, the method infers team attack directions from spatial distributions,
        assigns outlier players that were not seen during window processing, persists outputs, and notifies the backend.
        """
        player_features: Dict[int, np.ndarray] = {}
        player_id_to_bbox: Dict[int, List[float]] = {}
        crops_dir = os.path.join(os.path.dirname(self.assignment_output), "crops")
        os.makedirs(crops_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = 30
        window_size = 1800
        cluster_size = 2
        assigned_players = set()

        for window_start in range(0, total_frames, window_size):
            window_end = min(window_start + window_size, total_frames)
            first_occurrence: Dict[int, int] = {}

            # Identify the first frame index where each player appears within the current window.
            for frame_idx in range(window_start, window_end):
                if frame_idx % frame_step != 0:
                    continue
                frame_key = str(frame_idx)
                if frame_key not in self.tracking_log:
                    continue
                for player in self.tracking_log[frame_key].get("players", []):
                    pid = player["id"]
                    if pid in self.team_assignments or pid in assigned_players:
                        continue
                    if pid not in first_occurrence and "position_2d" in player:
                        first_occurrence[pid] = frame_idx

            # Prepare a minimal frame cache for all required first-occurrence frames.
            frame_cache: Dict[int, np.ndarray] = {}
            for f in sorted(set(first_occurrence.values())):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if ret:
                    frame_cache[f] = frame.copy()

            # Extract visual features from the cached first occurrences.
            for pid, frame_idx in first_occurrence.items():
                player_data = next(
                    (p for p in self.tracking_log[str(frame_idx)]["players"] if p["id"] == pid),
                    None
                )
                if player_data is None:
                    continue

                # Discard positions far from the core playing area in order to reduce noise.
                x, y = player_data["position_2d"]
                if not (0.15 * self.field_length_cm <= x <= 0.85 * self.field_length_cm and
                        0.1 * self.field_width_cm <= y <= 0.9 * self.field_width_cm):
                    continue

                x1, y1, x2, y2 = map(int, player_data["bbox"])
                crop = frame_cache[frame_idx][y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                feature = self._extract_cae_feature(crop)
                if feature is not None:
                    player_features[pid] = feature
                    player_id_to_bbox[pid] = player_data["bbox"]
                    assigned_players.add(pid)

            # Perform clustering for the current window when features are available.
            if player_features:
                self._cluster_and_assign(player_features, player_id_to_bbox, cluster_size, window_start, window_end)
                player_features.clear()
                player_id_to_bbox.clear()

        cap.release()
        self._infer_team_directions()
        self._assign_outlier_players(frame_step)
        self._save_team_assignment(total_frames)
        self._notify_backend()

    def _extract_cae_feature(self, image_crop) -> Optional[np.ndarray]:
        """
        Encodes a player crop into a compact appearance feature using the encoder submodel.
        The crop is resized to the network input resolution, normalized to the unit interval,
        and passed through the encoder. The flattened activation map is returned.
        The function returns None when the operation fails, for instance due to incompatible input shapes.
        """
        try:
            resized = cv2.resize(image_crop, (64, 64))
            norm = resized.astype("float32") / 255.0
            input_tensor = np.expand_dims(norm, axis=0)
            encoded = self.encoder.predict(input_tensor, verbose=0)
            return encoded.flatten()
        except Exception as e:
            print(f"[TeamAssignment] CAE feature extraction error: {e}")
            return None

    def _cluster_and_assign(
        self,
        player_features: dict,
        player_id_to_bbox: dict,
        cluster_size: int,
        window_start: int,
        window_end: int,
    ) -> None:
        """
        Reduces features by principal component analysis and performs KMeans clustering.
        The function selects the principal component dimension that maximizes the silhouette score up to a preset maximum.
        Cluster labels are mapped to team identifiers by descending cluster frequency.
        Diagnostic plots and weighted quality scores are recorded for later inspection.
        """
        features = list(player_features.values())
        player_ids = list(player_features.keys())

        if len(features) < 3:
            print("[TeamAssignment] Not enough features for clustering.")
            return

        best_dim, best_score, _ = self.evaluate_pca_silhouettes(features, max_components=10, cluster=cluster_size)
        if not best_dim:
            print("[TeamAssignment] No valid PCA dimension found.")
            return

        reduced = PCA(n_components=best_dim).fit_transform(features)
        labels = KMeans(n_clusters=cluster_size, random_state=42).fit_predict(reduced)

        dbi = davies_bouldin_score(reduced, labels)
        print(f"[TeamAssignment] Silhouette {best_score:.4f}, DBI {dbi:.4f}")

        self.silhouette_scores_weighted.append((best_score, len(player_ids)))
        self.dbi_scores_weighted.append((dbi, len(player_ids)))

        self._save_cluster_plot(features, player_ids, labels, window_start, window_end)

        # Map clusters to team ids by frequency.
        cluster_map = {c[0]: i + 1 for i, c in enumerate(Counter(labels).most_common()[:2])}
        for pid, label in zip(player_ids, labels):
            team_id = cluster_map.get(label, 0)
            if team_id:
                self.team_assignments[pid] = {
                    "team": team_id,
                    "bbox": player_id_to_bbox[pid],
                    "removed": False,
                }

    def _save_cluster_plot(self, features, player_ids, labels, window_start: int, window_end: int) -> None:
        """
        Saves a diagnostic scatter plot of the clustered features in a two-dimensional principal component projection.
        Player identifiers are printed near their projected points to support qualitative inspection of the clusters.
        The plot is stored in a session specific directory and named by the corresponding time window in minutes.
        """
        try:
            reduced_vis = PCA(n_components=2).fit_transform(features)
            plt.figure(figsize=(6, 5))
            for cluster_id in set(labels):
                points = reduced_vis[np.array(labels) == cluster_id]
                plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.7)
            for i, pid in enumerate(player_ids):
                plt.text(reduced_vis[i, 0], reduced_vis[i, 1], str(pid), fontsize=6, ha="center", va="center")
            plt.title("KMeans clustering of players")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.axis("equal")
            plt.legend()
            plt.tight_layout()

            out_dir = os.path.join(os.path.dirname(self.assignment_output), "cluster_plots")
            os.makedirs(out_dir, exist_ok=True)

            # At 30 frames per second, minutes are computed by frame_index divided by fps and by 60.
            start_min = window_start // 30 // 60
            end_min = window_end // 30 // 60
            plot_path = os.path.join(out_dir, f"cluster_plot_{start_min}min_to_{end_min}min_{len(player_ids)}players.png")

            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"[TeamAssignment] Saved cluster plot at {plot_path}")
        except Exception as e:
            print(f"[TeamAssignment] Cluster plot error: {e}")

    def evaluate_pca_silhouettes(self, features, max_components: int = 30, cluster: int = 2, metric: str = "euclidean"):
        """
        Evaluates the silhouette score over a range of principal component dimensions and returns the best dimension and score.
        The method fits principal component analysis for each dimension, clusters the reduced data with KMeans, and computes the silhouette.
        The function returns a tuple with the best dimension, the corresponding score, and a dictionary of all computed scores.
        """
        scores = {}
        for n in range(2, min(max_components, len(features[0])) + 1):
            try:
                reduced = PCA(n_components=n).fit_transform(features)
                labels = KMeans(n_clusters=cluster, random_state=42).fit_predict(reduced)
                if len(set(labels)) > 1:
                    scores[n] = silhouette_score(reduced, labels, metric=metric)
            except Exception as e:
                print(f"[TeamAssignment] PCA with {n} components raised an error: {e}")
        if scores:
            best_dim = max(scores, key=scores.get)
            return best_dim, scores[best_dim], scores
        return None, None, {}

    def _infer_team_directions(self) -> None:
        """
        Infers team attack directions from the distribution of x coordinates in the pitch reference frame.
        The method computes average x positions per team and assigns left to right and right to left
        based on the ordering of the averages. The default configuration is kept when insufficient evidence is available.
        """
        team_x = {"1": [], "2": []}
        for frame in self.tracking_log.values():
            for player in frame.get("players", []):
                pid = player["id"]
                team_info = self.team_assignments.get(pid)
                if team_info and "position_2d" in player:
                    team_x[str(team_info["team"])].append(player["position_2d"][0])
        avg_x = {team: np.mean(x) for team, x in team_x.items() if x}
        if len(avg_x) == 2:
            self.direction_config = (
                {"1": "left_to_right", "2": "right_to_left"}
                if avg_x["1"] < avg_x["2"]
                else {"1": "right_to_left", "2": "left_to_right"}
            )

    def _assign_outlier_players(self, frame_step: int) -> None:
        """
        Assigns team labels to players not covered during window-wise clustering.
        The method infers team membership from average horizontal positions sampled in regular frame intervals.
        The function records the assignment with the most probable team mapping and preserves the original bounding box.
        """
        all_pids = {p["id"] for f in self.tracking_log.values() for p in f.get("players", [])}
        for pid in all_pids:
            if pid in self.team_assignments:
                continue
            team_id = self._infer_outlier_team(pid, frame_step)
            if team_id:
                bbox = next(
                    (p["bbox"] for f in self.tracking_log.values() for p in f.get("players", []) if p["id"] == pid),
                    None,
                )
                if bbox:
                    self.team_assignments[pid] = {"team": team_id, "bbox": bbox, "removed": False}

    def _infer_outlier_team(self, pid: int, frame_step: int) -> Optional[int]:
        """
        Estimates the team of a single player by aggregating horizontal positions across sampled frames.
        The decision rule compares the mean x position to left and right corridors and combines this with
        the direction configuration in order to return the most likely team identifier.
        """
        x_positions = [
            p["position_2d"][0]
            for i in sorted(map(int, self.tracking_log))
            if i % frame_step == 0
            for p in self.tracking_log[str(i)].get("players", [])
            if p["id"] == pid and "position_2d" in p
        ]
        if not x_positions:
            return None
        avg_x = np.mean(x_positions)
        is_near_right = avg_x > 0.6 * self.field_length_cm
        is_near_left = avg_x < 0.4 * self.field_length_cm
        team1_right = self.direction_config["1"] == "right_to_left"
        return 1 if (is_near_right and team1_right) or (is_near_left and not team1_right) else 2

    def _save_team_assignment(self, total_frames: int) -> None:
        """
        Persists the team assignments and direction estimates to JSON and writes summary quality scores to a text file.
        The method also saves four evenly spaced frames from the source video as visual anchors for manual inspection.
        The function creates the necessary directories when they do not exist and does not alter the original inputs.
        """
        out_dir = os.path.dirname(self.assignment_output)
        out_json = {"players": self.team_assignments, "directions": self.direction_config}
        with open(self.assignment_output, "w") as f:
            json.dump(out_json, f, indent=4)

        sil_path = os.path.join(out_dir, "silhouette_DBI_scores.txt")
        sil = weighted_average(self.silhouette_scores_weighted) or 0.0
        dbi = weighted_average(self.dbi_scores_weighted) or 0.0
        with open(sil_path, "w") as f:
            f.write(f"Silhouette {sil:.4f}\n")
            f.write(f"DBI {dbi:.4f}\n")

        cap = cv2.VideoCapture(self.video_path)
        frame_dir = os.path.join(out_dir, "assignment_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i in [round(total_frames * x / 5) for x in range(1, 5)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.jpg"), frame)
        cap.release()

    def _notify_backend(self) -> None:
        """
        Sends the final team assignments and direction configuration to the backend.
        When manual assignment mode is enabled, the payload contains the session identifier and the team configuration
        and is posted to the notification endpoint. Otherwise the payload is posted to the default team assignment endpoint.
        The method logs a success message for HTTP status code two hundred and prints the status code otherwise.
        """
        payload = {"players": self.team_assignments, "directions": self.direction_config}
        if self.run_manual_assignment:
            payload.update({"session_id": self.session_id, "team_config": self.team_config})
            endpoint = f"{self.session_id}/team-assignment/notify"
        else:
            endpoint = f"{self.session_id}/team-assignment"
        try:
            r = requests.post(self.api_url(endpoint), json=payload)
            print("[TeamAssignment] Web UI notified." if r.status_code == 200 else f"[TeamAssignment] Notification failed with status {r.status_code}.")
        except Exception as e:
            print(f"[TeamAssignment] Notification error: {e}")
