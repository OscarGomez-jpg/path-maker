import cv2
import numpy as np
from ultralytics import FastSAM


class DepthProcessor:
    """Processor class for robotic navigation using FastSAM segmentation.

    This class handles the conversion of raw camera frames into segmented objects,
    identifies the walkable floor, and calculates a safe path for a robot.
    """

    def __init__(self):
        """Initializes the FastSAM model and sets the execution device."""
        self.model = FastSAM("FastSAM-s.pt")
        self.device = "cpu"
        print(f"Robot Navigator (FastSAM) loaded on {self.device}")

    def apply_preprocessing(self, frame, gamma=1.2, clahe_clip=2.0):
        """Enhances image visibility to improve segmentation quality.

        Args:
            frame: The raw BGR image from the camera.
            gamma: Mid-tone correction value.
            clahe_clip: Contrast Limited Adaptive Histogram Equalization clip limit.

        Returns:
            The pre-processed BGR image.
        """
        # Gamma Correction
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        img = cv2.LUT(frame, table)

        # LAB Contrast Enhancement (Preserves colors)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img = cv2.merge((cl, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        return img

    def process_frame(self, frame, params):
        """Identifies objects, floor, and calculates a safe navigation path.

        Args:
            frame: Raw BGR image.
            params: Dictionary containing 'min_area' for object filtering.

        Returns:
            tuple: (points_data, path_points, debug_img)
                - points_data: List of dicts containing object contours and types.
                - path_points: List of [x, y] coordinates for the safe path.
                - debug_img: Processed image with overlays for monitoring.
        """
        min_area = params.get("min_area", 1000)
        h, w = frame.shape[:2]

        # FastSAM Inference
        # everything=True allows segmenting all objects without specific prompts
        results = self.model(
            frame, device=self.device, retina_masks=True, imgsz=320, conf=0.4, iou=0.9
        )[0]

        points_data = []
        path_points = []
        floor_mask = np.zeros((h, w), dtype=np.uint8)

        if results.masks is not None:
            # Floor Identification Heuristic:
            # We assume the robot starts at the bottom-center of the frame.
            start_point = (w // 2, h - 20)
            masks_data = results.masks.data.cpu().numpy()

            floor_idx = -1
            for i, mask in enumerate(masks_data):
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Check if the mask covers the robot's starting area
                if mask[start_point[1], start_point[0]] > 0:
                    floor_idx = i
                    floor_mask = (mask * 255).astype(np.uint8)
                    break

            # Obstacles and floor contours
            for i, mask_pts in enumerate(results.masks.xy):
                pts = np.array(mask_pts, dtype=np.int32)
                if cv2.contourArea(pts) < min_area:
                    continue

                if i != floor_idx:
                    points_data.append(
                        {
                            "type": "obstacle",
                            "points": pts[::3].tolist(),
                            "color": [0, 0, 255],
                        }
                    )
                else:
                    points_data.append(
                        {
                            "type": "floor",
                            "points": pts[::4].tolist(),
                            "color": [0, 255, 0],
                        }
                    )

            # PATHFINDING CALCULATION
            if floor_idx != -1:
                # Morphological Erosion: Shrink floor to create a safety buffer around obstacles.
                # This ensures the robot center doesn't get too close to walls/objects.
                kernel = np.ones((15, 15), np.uint8)
                safe_floor = cv2.erode(floor_mask, kernel, iterations=1)

                # Scan-line approach: Look for horizontal gaps from bottom to top
                for y in range(h - 20, h // 3, -40):
                    row = safe_floor[y, :]
                    on_segments = np.where(row > 0)[0]

                    if len(on_segments) > 0:
                        # Group continuous floor pixels into segments
                        diff = np.diff(on_segments)
                        splits = np.where(diff > 1)[0] + 1
                        segments = np.split(on_segments, splits)

                        # Select the widest segment and target its center
                        best_seg = max(segments, key=len)
                        target_x = int(np.mean(best_seg))
                        path_points.append([target_x, y])

        # Generate Debug Visualization
        debug_img = results.plot()
        if len(path_points) > 1:
            pts_array = np.array(path_points, np.int32).reshape((-1, 1, 2))
            # Draw the Cyan navigation path
            cv2.polylines(debug_img, [pts_array], False, (255, 255, 0), 5)
            for p in path_points:
                cv2.circle(debug_img, tuple(p), 8, (255, 255, 255), -1)

        return points_data, path_points, debug_img
