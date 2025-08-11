import numpy as np
import supervision as sv


class PlayerAnnotator:
    """
    Annotates detected players in a video frame using team-specific color configurations.
    Players are drawn as ellipses at their bounding box locations, with colors determined
    by their assigned team.
    """

    def __init__(self, team_config):
        """
        Initializes the player annotator using the given team color configuration.

        Parameters
        ----------
        team_config : dict
            Dictionary defining team-specific colors and other visual settings.
            Expected format:
            {
                "1": {"color": "#HEXCODE"},
                "2": {"color": "#HEXCODE"}
            }
        """
        self.team_config = team_config

        team_colors = sv.ColorPalette([
            sv.Color.from_hex(self.team_config["1"]["color"]),
            sv.Color.from_hex(self.team_config["2"]["color"])
        ])

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=team_colors,
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS
        )

        # Optional label annotator for drawing player IDs or names (currently disabled)
        # self.label_annotator = sv.LabelAnnotator(
        #     color=team_colors,
        #     text_color=sv.Color.from_hex("#FFFFFF"),
        #     text_position=sv.Position.BOTTOM_CENTER,
        #     color_lookup=sv.ColorLookup.CLASS
        # )

    def annotate(self, frame, player_data, pitch_coords_dict, team_assignments):
        """
        Annotates players on the given frame with ellipses corresponding to their team.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame in BGR format.
        player_data : list[dict]
            List of player dictionaries containing at least "id" and "bbox".
        pitch_coords_dict : dict[int, np.ndarray]
            Mapping of player IDs to their 2D pitch coordinates in metric units.
        team_assignments : dict[int, int]
            Mapping of player IDs to their assigned team number (1 or 2).

        Returns
        -------
        tuple
            annotated_frame : np.ndarray
                The frame with player ellipses drawn.
            player_coords_team_dict : dict
                Mapping from player ID to a dictionary containing:
                    "position_2d" : np.ndarray
                        The player's position in pitch coordinates.
                    "team" : int
                        Team index (0 for team 1, 1 for team 2).
        """
        boxes, class_ids, player_ids = [], [], []
        player_coords_team_dict = {}
        annotated_frame = frame.copy()

        # Map player IDs to bounding boxes for quick access
        id_to_bbox = {int(p["id"]): p["bbox"] for p in player_data if "bbox" in p}

        for pid, coords in pitch_coords_dict.items():
            team = team_assignments.get(pid)
            if team not in (1, 2):
                continue

            if pid not in id_to_bbox:
                continue

            box = id_to_bbox[pid]
            x1, y1, x2, y2 = box
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                print(f"[PlayerAnnotator] Invalid bounding box for ID {pid} – skipped.")
                continue

            cls_id = 0 if team == 1 else 1
            player_coords_team_dict[pid] = {"position_2d": coords, "team": cls_id}
            boxes.append(box)
            class_ids.append(cls_id)
            player_ids.append(pid)

        # Draw ellipses for all valid player detections
        if boxes:
            detections = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            annotated_frame = self.ellipse_annotator.annotate(scene=annotated_frame, detections=detections)

        return annotated_frame, player_coords_team_dict
