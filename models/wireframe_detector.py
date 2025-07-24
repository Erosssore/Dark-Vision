import cv2
import time

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

class YOLOv8PoseDetector:
    """
    YOLOv8-based detector for human detection with wireframe visualization.
    This class detects human figures and renders pose overlays on them.
    """

    def __init__(self, confidence_threshold=0.5, model_path="yolov8n-pose.pt"):
        """
        Initialize the YOLOv8 pose detector

        Args:
            confidence_threshold: Minimum confidence score for human detection (0.0-1.0)
            model_path: Path to YOLOv8-pose weights (or model name, e.g., 'yolov8n-pose.pt')
        """
        # Load YOLOv8-pose model
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

        # Processing stats
        self.process_time = 0

        # COCO skeleton pairs (for 17-keypoint model)
        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # right arm
            (0, 5), (5, 6), (6, 7), (7, 8),      # left arm
            (0, 9), (9, 10), (10, 11), (11, 12), # right leg
            (0, 13), (13, 14), (14, 15), (15, 16) # left leg
        ]

    def set_confidence_threshold(self, threshold):
        """
        Set the confidence threshold for detections

        Args:
            threshold: Value between 0.0 and 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def toggle(self):
        """Toggle detection on/off"""
        self.enabled = not self.enabled
        return self.enabled

    def toggle_wireframe(self):
        """Toggle wireframe display on/off"""
        self.show_wireframe = not self.show_wireframe
        return self.show_wireframe

    def toggle_boxes(self):
        """Toggle bounding box display on/off"""
        self.show_boxes = not self.show_boxes
        return self.show_boxes

    def detect(self, frame):
        """
        Run YOLOv8 pose detection on the frame

        Args:
            frame: Image to process (numpy array)

        Returns:
            boxes: List of bounding boxes (x, y, w, h)
            confidences: List of confidence scores for each box
            keypoints_list: List of keypoints arrays for each detection
            process_time: Time taken to process the frame
        """
        if not self.enabled:
            self.process_time = 0
            return [], [], [], 0

        start_time = time.time()
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        boxes = []
        confidences = []
        keypoints_list = []

        for result in results:
            for box, kpts, conf in zip(result.boxes.xyxy.cpu().numpy() if result.boxes else [],
                                       result.keypoints.xy.cpu().numpy() if result.keypoints else [],
                                       result.boxes.conf.cpu().numpy() if result.boxes else []):
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                boxes.append([x1, y1, w, h])
                confidences.append(float(conf))
                keypoints_list.append(kpts.astype(int))  # shape: (17, 2)

        self.process_time = time.time() - start_time
        return boxes, confidences, keypoints_list, self.process_time

    def draw_wireframe(self, frame, keypoints):
        """
        Draw a human wireframe skeleton using detected keypoints

        Args:
            frame: Image to draw on
            keypoints: Array of keypoint coordinates (num_keypoints, 2)
        """
        # Draw skeleton connections
        for i, j in self.skeleton_connections:
            if i < len(keypoints) and j < len(keypoints):
                pt1, pt2 = tuple(keypoints[i]), tuple(keypoints[j])
                if all(pt1) and all(pt2):  # skip zero points
                    cv2.line(frame, pt1, pt2, self.wireframe_color, self.wireframe_thickness)
        # Draw keypoints (small circles)
        for pt in keypoints:
            if all(pt):  # skip zero points
                cv2.circle(frame, tuple(pt), self.kpt_radius, self.kpt_color, -1)

    def process(self, frame):
        """
        Process a frame for person detection and wireframe visualization

        Args:
            frame: Image to process

        Returns:
            annotated_frame: Frame with detections and wireframes visualized
            result_dict: Dictionary with detection results
        """
        if not self.enabled:
            return frame, {"humans_detected": 0, "boxes": [], "process_time": 0}

        annotated_frame = frame.copy()
        boxes, confidences, keypoints_list, process_time = self.detect(frame)

        for i, (box, conf, kpts) in enumerate(zip(boxes, confidences, keypoints_list)):
            x, y, w, h = box
            if self.show_boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
                label = f"Human: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.box_color, 2)
            if self.show_wireframe:
                self.draw_wireframe(annotated_frame, kpts)

        cv2.putText(
            annotated_frame,
            f"Humans: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Process Time: {process_time * 1000:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        result_dict = {
            "humans_detected": len(boxes),
            "boxes": boxes,
            "confidences": confidences,
            "process_time": process_time
        }
        return annotated_frame, result_dict