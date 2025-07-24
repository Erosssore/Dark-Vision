import cv2
import numpy as np
import os
import time


class YOLODetector:
    """
    YOLOv4-tiny detector for human detection with wireframe visualization.
    This class detects human figures and renders wireframe overlays on them.
    Enhanced with optical flow tracking for improved motion accuracy.
    """

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the YOLO detector

        Args:
            confidence_threshold: Minimum confidence score for human detection (0.0-1.0)
        """
        # Get the path to the dataset directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        yolo_dir = os.path.join(project_root, "datasets")

        # Paths to YOLO files
        self.config_path = os.path.join(yolo_dir, "yolov4-tiny.cfg")
        self.weights_path = os.path.join(yolo_dir, "yolov4-tiny.weights")

        # Verify files exist
        if not os.path.exists(self.config_path) or not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"YOLO model files not found at {self.config_path} or {self.weights_path}.\n"
                "Please make sure the yolov4-tiny.cfg and yolov4-tiny.weights files "
                "are in the datasets folder."
            )

        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

        # Set preferred backend and target for better performance
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.5.4+
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # Older OpenCV versions
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Detection parameters
        self.conf_threshold = confidence_threshold
        self.nms_threshold = 0.4
        self.input_size = (416, 416)

        # Person is class 0 in COCO dataset
        self.person_class_id = 0

        # Processing stats
        self.process_time = 0

        # Wireframe styling
        self.wireframe_color = (0, 255, 255)  # Cyan color for wireframe
        self.wireframe_thickness = 2

        # Human skeleton keypoints (relative positions within bounding box)
        # Format: [(x_ratio, y_ratio), ...] where ratios are relative to bounding box dimensions
        self.skeleton_keypoints = {
            "head": (0.5, 0.15),
            "neck": (0.5, 0.25),
            "right_shoulder": (0.7, 0.25),
            "left_shoulder": (0.3, 0.25),
            "right_elbow": (0.8, 0.4),
            "left_elbow": (0.2, 0.4),
            "right_hand": (0.85, 0.55),
            "left_hand": (0.15, 0.55),
            "torso": (0.5, 0.5),
            "right_hip": (0.6, 0.65),
            "left_hip": (0.4, 0.65),
            "right_knee": (0.65, 0.8),
            "left_knee": (0.35, 0.8),
            "right_foot": (0.65, 0.95),
            "left_foot": (0.35, 0.95)
        }

        # Skeleton connections (which points should be connected with lines)
        self.skeleton_connections = [
            ("head", "neck"),
            ("neck", "right_shoulder"),
            ("neck", "left_shoulder"),
            ("right_shoulder", "right_elbow"),
            ("left_shoulder", "left_elbow"),
            ("right_elbow", "right_hand"),
            ("left_elbow", "left_hand"),
            ("neck", "torso"),
            ("torso", "right_hip"),
            ("torso", "left_hip"),
            ("right_hip", "right_knee"),
            ("left_hip", "left_knee"),
            ("right_knee", "right_foot"),
            ("left_knee", "left_foot")
        ]

        # Additional detection parameters
        self.use_gpu = False
        self.enabled = True
        self.show_wireframe = True
        self.show_boxes = True

        # NEW: Tracking parameters
        self.prev_frame = None
        self.prev_gray = None
        self.prev_boxes = []
        self.prev_keypoints = {}
        self.tracking_enabled = True
        self.detection_interval = 5  # Run full detection every N frames
        self.frame_count = 0
        self.keypoint_history = {}  # For temporal smoothing
        self.keypoint_history_length = 3

        # NEW: Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # NEW: Box tracking parameters
        self.tracker = None
        self.tracker_types = ['KCF', 'CSRT']
        self.current_tracker = 'KCF'

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

    def toggle_gpu(self):
        """Toggle between CPU and GPU processing if available"""
        if self.use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_gpu = False
            print("Using CPU for processing")
        else:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.use_gpu = True
                print("Using GPU for processing")
            except:
                print("GPU acceleration not available")
                self.use_gpu = False

    def toggle_tracking(self):
        """Toggle optical flow tracking on/off"""
        self.tracking_enabled = not self.tracking_enabled
        return self.tracking_enabled

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

    def create_tracker(self, tracker_type='KCF'):
        """Create a tracker instance based on tracker type"""
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            return cv2.TrackerKCF_create()  # Default

    def detect(self, frame):
        """
        Run YOLO detection on the frame

        Args:
            frame: Image to process (numpy array)

        Returns:
            boxes: List of bounding boxes (x, y, w, h)
            confidences: List of confidence scores for each box
            process_time: Time taken to process the frame
        """
        if not self.enabled:
            self.process_time = 0
            return [], [], 0

        start_time = time.time()

        # Get image dimensions
        height, width = frame.shape[:2]

        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Forward pass through the network
        outputs = self.net.forward(self.output_layers)

        # Process detections
        boxes = []
        confidences = []

        # Process detections from each output layer
        for output in outputs:
            for detection in output:
                # YOLO format: [center_x, center_y, width, height, confidence, class_scores...]
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(detection[4])

                # Only process person class with enough confidence
                if class_id == self.person_class_id and confidence > self.conf_threshold:
                    # Convert YOLO box coordinates to regular format
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

        # Apply non-maximum suppression to remove overlapping boxes
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.conf_threshold, self.nms_threshold
            )

            # Extract results after NMS
            result_boxes = []
            result_confidences = []

            if len(indices) > 0:
                # OpenCV 4.5.4+ returns a flat array
                if isinstance(indices, np.ndarray):
                    for idx in indices.flatten():
                        result_boxes.append(boxes[idx])
                        result_confidences.append(confidences[idx])
                # Older OpenCV versions return a nested array
                else:
                    for idx in indices:
                        i = idx[0] if isinstance(idx, list) else idx
                        result_boxes.append(boxes[i])
                        result_confidences.append(confidences[i])
        else:
            result_boxes = []
            result_confidences = []

        self.process_time = time.time() - start_time

        return result_boxes, result_confidences, self.process_time

    def track_boxes(self, frame, prev_boxes):
        """
        Track bounding boxes using optical flow

        Args:
            frame: Current frame
            prev_boxes: List of bounding boxes from previous frame

        Returns:
            tracked_boxes: Updated bounding box positions
        """
        if not prev_boxes or self.prev_gray is None:
            return []

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For each box, track its corners using optical flow
        tracked_boxes = []

        for box in prev_boxes:
            x, y, w, h = box

            # Define box corners for tracking
            pts = np.float32([
                [x, y],  # top-left
                [x + w, y],  # top-right
                [x, y + h],  # bottom-left
                [x + w, y + h]  # bottom-right
            ]).reshape(-1, 1, 2)

            # Calculate optical flow
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None, **self.lk_params
            )

            # Filter good points
            if status is not None and np.sum(status) >= 3:  # At least 3 corners tracked
                good_new = new_pts[status.flatten() == 1]

                # Calculate new bounding box from tracked points
                x_min = np.min(good_new[:, 0])
                y_min = np.min(good_new[:, 1])
                x_max = np.max(good_new[:, 0])
                y_max = np.max(good_new[:, 1])

                new_x = int(x_min)
                new_y = int(y_min)
                new_w = int(x_max - x_min)
                new_h = int(y_max - y_min)

                # Add tracked box
                tracked_boxes.append([new_x, new_y, new_w, new_h])

        # Update previous frame
        self.prev_gray = gray.copy()

        return tracked_boxes

    def calculate_keypoints(self, box):
        """
        Calculate keypoint positions based on the bounding box

        Args:
            box: Bounding box coordinates [x, y, w, h]

        Returns:
            Dictionary of keypoints
        """
        x, y, w, h = box
        keypoints = {}

        for name, (x_ratio, y_ratio) in self.skeleton_keypoints.items():
            px = int(x + x_ratio * w)
            py = int(y + y_ratio * h)
            keypoints[name] = (px, py)

        return keypoints

    def track_keypoints(self, frame, prev_keypoints):
        """
        Track keypoints using optical flow

        Args:
            frame: Current frame
            prev_keypoints: Dictionary of keypoints from previous frame

        Returns:
            tracked_keypoints: Updated keypoint positions
        """
        if not prev_keypoints or self.prev_gray is None:
            return {}

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert keypoints dictionary to list of points for tracking
        keypoint_names = list(prev_keypoints.keys())
        points = np.array([prev_keypoints[name] for name in keypoint_names], dtype=np.float32).reshape(-1, 1, 2)

        # Calculate optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, points, None, **self.lk_params
        )

        # Create new keypoints dictionary
        tracked_keypoints = {}

        for i, name in enumerate(keypoint_names):
            # Check if status is a 2D array or a 1D array
            if isinstance(status[i], np.ndarray):
                is_tracked = status[i][0] == 1
            else:
                is_tracked = status[i] == 1

            if is_tracked:  # If point was tracked successfully
                tracked_keypoints[name] = (int(new_points[i][0][0]), int(new_points[i][0][1]))
            else:
                # If tracking failed, keep original position
                tracked_keypoints[name] = prev_keypoints[name]

        # Update previous frame
        self.prev_gray = gray.copy()

        return tracked_keypoints

    def smooth_keypoints(self, current_keypoints, person_id):
        """
        Apply temporal smoothing to keypoints

        Args:
            current_keypoints: Dictionary of current keypoint positions
            person_id: ID of the person to track across frames

        Returns:
            smoothed_keypoints: Keypoints after temporal smoothing
        """
        # Initialize history for this person if needed
        if person_id not in self.keypoint_history:
            self.keypoint_history[person_id] = []

        # Add current keypoints to history
        self.keypoint_history[person_id].append(current_keypoints)

        # Keep history at desired length
        if len(self.keypoint_history[person_id]) > self.keypoint_history_length:
            self.keypoint_history[person_id].pop(0)

        # Apply smoothing only if we have enough history
        if len(self.keypoint_history[person_id]) < 2:
            return current_keypoints

        # Create smoothed keypoints by averaging positions
        smoothed_keypoints = {}

        for name in current_keypoints.keys():
            # Get all positions of this keypoint in history
            positions = [frame_keypoints.get(name, (0, 0))
                         for frame_keypoints in self.keypoint_history[person_id]
                         if name in frame_keypoints]

            if positions:
                # Calculate weighted average, giving more weight to recent positions
                weights = np.linspace(0.5, 1.0, len(positions))
                weighted_x = sum(w * p[0] for w, p in zip(weights, positions)) / sum(weights)
                weighted_y = sum(w * p[1] for w, p in zip(weights, positions)) / sum(weights)

                smoothed_keypoints[name] = (int(weighted_x), int(weighted_y))
            else:
                # If no history, use current position
                smoothed_keypoints[name] = current_keypoints[name]

        return smoothed_keypoints

    def draw_wireframe(self, frame, keypoints):
        """
        Draw a human wireframe skeleton using the provided keypoints

        Args:
            frame: Image to draw on
            keypoints: Dictionary of keypoint positions
        """
        # Draw skeleton connections
        for start_point, end_point in self.skeleton_connections:
            if start_point in keypoints and end_point in keypoints:
                cv2.line(
                    frame,
                    keypoints[start_point],
                    keypoints[end_point],
                    self.wireframe_color,
                    self.wireframe_thickness
                )

        # Draw keypoints (small circles)
        for point in keypoints.values():
            cv2.circle(frame, point, 3, self.wireframe_color, -1)

    def process(self, frame):
        """
        Process a frame for person detection and wireframe visualization
        Enhanced with optical flow tracking for improved motion accuracy

        Args:
            frame: Image to process

        Returns:
            annotated_frame: Frame with detections and wireframes visualized
            result_dict: Dictionary with detection results
        """
        if not self.enabled:
            return frame, {"humans_detected": 0, "boxes": [], "process_time": 0}

        # Create output copy
        annotated_frame = frame.copy()

        # Convert frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decide whether to run detection or tracking
        run_detection = (self.frame_count % self.detection_interval == 0) or not self.prev_boxes

        if run_detection:
            # Run full detection
            start_time = time.time()
            boxes, confidences, _ = self.detect(frame)
            self.process_time = time.time() - start_time

            # Reset tracking data with new detections
            self.prev_boxes = boxes.copy() if boxes else []

            # Clear keypoint history when new detections occur
            self.keypoint_history = {}
        else:
            # Use tracking instead of detection
            start_time = time.time()

            if self.tracking_enabled and self.prev_gray is not None:
                # Track boxes using optical flow
                boxes = self.track_boxes(frame, self.prev_boxes)

                # If tracking failed, run detection
                if not boxes:
                    boxes, confidences, _ = self.detect(frame)
                    self.prev_boxes = boxes.copy() if boxes else []
                else:
                    # Use previous confidences for tracked boxes
                    confidences = [0.8] * len(boxes)  # Assign reasonable confidence to tracked boxes
                    self.prev_boxes = boxes.copy()
            else:
                # If tracking disabled, run detection
                boxes, confidences, _ = self.detect(frame)
                self.prev_boxes = boxes.copy() if boxes else []

            self.process_time = time.time() - start_time

        # Update frame counter
        self.frame_count += 1

        # Draw bounding boxes and wireframes
        for i, box in enumerate(boxes):
            person_id = i  # Simple ID assignment

            # Calculate or track keypoints
            if run_detection or not self.tracking_enabled:
                # Calculate keypoints based on box dimensions
                keypoints = self.calculate_keypoints(box)
            else:
                # If we have previous keypoints for this person, track them
                prev_person_keypoints = self.prev_keypoints.get(person_id, None)

                if prev_person_keypoints and self.prev_gray is not None:
                    # Track keypoints using optical flow
                    keypoints = self.track_keypoints(frame, prev_person_keypoints)
                else:
                    # Calculate keypoints if no previous data
                    keypoints = self.calculate_keypoints(box)

            # Apply temporal smoothing to keypoints
            keypoints = self.smooth_keypoints(keypoints, person_id)

            # Store keypoints for next frame
            if not hasattr(self, 'prev_keypoints'):
                self.prev_keypoints = {}
            self.prev_keypoints[person_id] = keypoints

            # Draw bounding box if enabled
            if self.show_boxes:
                x, y, w, h = box
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Add confidence label if available
                if i < len(confidences):
                    label = f"Human: {confidences[i]:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # Draw wireframe if enabled
            if self.show_wireframe:
                self.draw_wireframe(annotated_frame, keypoints)

        # Update previous frame data
        self.prev_gray = gray.copy()

        # Add status information
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
            f"Process Time: {self.process_time * 1000:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Add tracking status
        tracking_status = "Tracking ON" if self.tracking_enabled else "Tracking OFF"
        cv2.putText(
            annotated_frame,
            tracking_status,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Create result dictionary with all relevant information
        result_dict = {
            "humans_detected": len(boxes),
            "boxes": boxes,
            "confidences": confidences if len(confidences) == len(boxes) else [0.8] * len(boxes),
            "process_time": self.process_time,
            "tracking_enabled": self.tracking_enabled,
            "frame_mode": "Detection" if run_detection else "Tracking"
        }

        return annotated_frame, result_dict
