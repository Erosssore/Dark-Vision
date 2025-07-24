import cv2
import time
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

class YOLOv8PoseDetector:
    """
    YOLOv8-based detector for human detection with wireframe visualization.
    This class detects human figures and renders pose overlays on them.
    """

    def __init__(self, confidence_threshold=0.5, model_path="yolov8n-pose.pt", device=None):
        """
        Initialize the YOLOv8 pose detector

        Args:
            confidence_threshold: Minimum confidence score for human detection (0.0-1.0)
            model_path: Path to YOLOv8-pose weights (or model name, e.g., 'yolov8n-pose.pt')
        """
        # Device selection
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, "has_mps") and torch.has_mps:
                self.device = "mps"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            elif hasattr(torch.backends, "hip") and torch.backends.hip.is_built():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"YOLOv8PoseDetector using device: {self.device}")

        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold

        # Display settings
        self.enabled = True
        self.show_wireframe = True
        self.show_boxes = True

        # Colors and style
        self.wireframe_color = (0, 255, 255)
        self.wireframe_thickness = 2
        self.box_color = (0, 255, 0)
        self.box_thickness = 2
        self.kpt_color = (0, 255, 0)
        self.kpt_radius = 3

        # COCO skeleton pairs (for 17-keypoint model, YOLOv8 order)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),          # Nose to eyes/ears
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Shoulders, arms
            (5, 6),                                  # Shoulder line
            (5, 11), (6, 12), (11, 12),              # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
        ]

    def set_confidence_threshold(self, threshold):
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def toggle_wireframe(self):
        self.show_wireframe = not self.show_wireframe
        return self.show_wireframe

    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        return self.show_boxes

    def detect(self, frame):
        """
        Run YOLOv8 pose detection on the frame

        Returns:
            boxes: List of bounding boxes (x, y, w, h)
            confidences: List of confidence scores for each box
            keypoints_list: List of keypoints arrays for each detection
            keypoint_confs_list: List of confidences arrays for each detection
            process_time: Time taken to process the frame
        """
        if not self.enabled:
            return [], [], [], [], 0

        start_time = time.time()
        results = self.model.predict(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        boxes = []
        confidences = []
        keypoints_list = []
        keypoint_confs_list = []

        for result in results:
            if (result.boxes is not None and result.keypoints is not None):
                box_arr = result.boxes.xyxy.cpu().numpy()
                conf_arr = result.boxes.conf.cpu().numpy()
                kpts_arr = result.keypoints.xy.cpu().numpy()
                kpts_conf_arr = result.keypoints.conf.cpu().numpy()
                for box, kpts, conf, kpt_confs in zip(box_arr, kpts_arr, conf_arr, kpts_conf_arr):
                    x1, y1, x2, y2 = box.astype(int)
                    w, h = x2 - x1, y2 - y1
                    boxes.append([x1, y1, w, h])
                    confidences.append(float(conf))
                    keypoints_list.append(kpts)
                    keypoint_confs_list.append(kpt_confs)
        process_time = time.time() - start_time
        return boxes, confidences, keypoints_list, keypoint_confs_list, process_time

    def draw_wireframe(self, frame, keypoints, keypoint_confs, conf_thresh=0.3):
        """
        Draw a human wireframe skeleton using detected keypoints and confidences

        Args:
            frame: Image to draw on
            keypoints: Array of keypoint coordinates (num_keypoints, 2)
            keypoint_confs: Array of confidence scores for each keypoint
            conf_thresh: Confidence threshold for drawing joints/limbs
        """
        # Draw skeleton connections
        for i, j in self.skeleton_connections:
            if (
                i < len(keypoints) and j < len(keypoints) and
                keypoint_confs[i] > conf_thresh and keypoint_confs[j] > conf_thresh
            ):
                pt1 = tuple(map(int, keypoints[i]))
                pt2 = tuple(map(int, keypoints[j]))
                cv2.line(frame, pt1, pt2, self.wireframe_color, self.wireframe_thickness)
        # Draw keypoints
        for idx, pt in enumerate(keypoints):
            if keypoint_confs[idx] > conf_thresh:
                cv2.circle(frame, tuple(map(int, pt)), self.kpt_radius, self.kpt_color, -1)

    def process(self, frame):
        if not self.enabled:
            return frame, {"humans_detected": 0, "boxes": [], "process_time": 0}

        annotated_frame = frame.copy()
        boxes, confidences, keypoints_list, keypoint_confs_list, process_time = self.detect(frame)

        for box, conf, kpts, kpt_confs in zip(boxes, confidences, keypoints_list, keypoint_confs_list):
            x, y, w, h = box
            if self.show_boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
                label = f"Human: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, self.box_color, 2)
            if self.show_wireframe:
                self.draw_wireframe(annotated_frame, kpts, kpt_confs)

        result_dict = {
            "humans_detected": len(boxes),
            "boxes": boxes,
            "confidences": confidences,
            "process_time": process_time
        }

        return annotated_frame, result_dict