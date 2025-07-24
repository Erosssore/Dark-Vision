import cv2
import time
import tkinter as tk
from tkinter import filedialog


class VideoPlayer:
    def __init__(self):
        """Initialize the video player with default settings"""
        # Video source
        self.cap = None

        # Processing components
        self.enhancer = None
        self.yolo_detector = None

        # Video properties
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        # Playback state
        self.playing = False
        self.current_frame = None
        self.processed_frame = None

        # Feature toggles
        self.overlay_enabled = True
        self.enhancer_enabled = True
        self.detector_enabled = True

        # Performance tracking
        self.last_frame_time = 0
        self.display_fps = 0

        print("Video Player initialized")

    def set_yolo_detector(self, detector):
        """Set the YOLO detector for human detection"""
        self.yolo_detector = detector
        if detector is not None:
            self.detector_enabled = True
            print("YOLO detector attached")
        else:
            self.detector_enabled = False
            print("YOLO detector disabled")

    def set_enhancer(self, enhancer):
        """Set the video enhancer for low-light enhancement"""
        self.enhancer = enhancer
        if enhancer is not None:
            self.enhancer_enabled = True
            print("Video enhancer attached")
        else:
            self.enhancer_enabled = False
            print("Video enhancer disabled")

    def toggle_enhancer(self):
        """Toggle the video enhancer on/off"""
        if self.enhancer is not None:
            self.enhancer_enabled = not self.enhancer_enabled
            print(f"Video enhancement: {'ON' if self.enhancer_enabled else 'OFF'}")
        else:
            print("No video enhancer available")

    def toggle_detector(self):
        """Toggle the human detector on/off"""
        if self.yolo_detector is not None:
            self.detector_enabled = not self.detector_enabled
            print(f"Human detection: {'ON' if self.detector_enabled else 'OFF'}")
        else:
            print("No detector available")

    def toggle_wireframe(self):
        """Toggle wireframe visualization for detected humans"""
        if self.yolo_detector is not None:
            self.yolo_detector.toggle_wireframe()
            print(f"Wireframe visualization: {'ON' if self.yolo_detector.show_wireframe else 'OFF'}")
        else:
            print("No detector available")

    def toggle_boxes(self):
        """Toggle bounding box visualization for detected humans"""
        if self.yolo_detector is not None:
            self.yolo_detector.toggle_boxes()
            print(f"Bounding boxes: {'ON' if self.yolo_detector.show_boxes else 'OFF'}")
        else:
            print("No detector available")

    def toggle_overlay(self):
        """Toggle information overlay display"""
        self.overlay_enabled = not self.overlay_enabled
        print(f"Information overlay: {'ON' if self.overlay_enabled else 'OFF'}")

    def toggle_gpu(self):
        """Toggle GPU acceleration for processing if available"""
        if self.enhancer is not None:
            self.enhancer.toggle_gpu()

        if self.yolo_detector is not None:
            self.yolo_detector.toggle_gpu()

    def load_video(self, video_path):
        """Load a video file from the specified path"""
        try:
            # Open the video file
            self.cap = cv2.VideoCapture(video_path)

            # Check if the video was opened successfully
            if not self.cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return False

            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Reset frame counter
            self.frame_count = 0

            # Print video information
            print(f"Loaded video: {video_path}")
            print(f"Resolution: {self.width}x{self.height}")
            print(f"FPS: {self.fps:.2f}")
            print(f"Total frames: {total_frames}")
            print(f"Duration: {total_frames / self.fps:.2f} seconds")

            return True

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def select_video_file(self):
        """Open a file dialog to select a video file"""
        # Create a temporary Tkinter root window for the file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open file dialog
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )

        # Destroy the temporary root window
        root.destroy()

        # If a file was selected, load it
        if video_path:
            return self.load_video(video_path)
        else:
            print("No file selected")
            return False

    def process_frame(self, frame):
        """Process a single frame through the enhancement and detection pipeline"""
        if frame is None:
            return None

        start_time = time.time()
        processing_stats = {}
        enhanced_frame = frame.copy()

        # Step 1: Apply video enhancement if enabled
        if self.enhancer is not None and self.enhancer_enabled:
            try:
                enhanced_frame, enhancer_stats = self.enhancer.process(frame)
                processing_stats.update(enhancer_stats)
            except Exception as e:
                print(f"Error in enhancement processing: {e}")
                enhanced_frame = frame.copy()  # Fall back to original frame

        # Step 2: Apply human detection if enabled
        detected_frame = enhanced_frame.copy()
        if self.yolo_detector is not None and self.detector_enabled:
            try:
                detected_frame, detector_stats = self.yolo_detector.process(enhanced_frame)
                processing_stats.update(detector_stats)
            except Exception as e:
                print(f"Error in detection processing: {e}")
                detected_frame = enhanced_frame  # Fall back to enhanced frame

        # Step 3: Create information overlay if enabled
        final_frame = detected_frame
        if self.overlay_enabled:
            # Make sure all required keys are present in processing_stats
            required_stats = {
                'frame_count': self.frame_count,
                'original_brightness': 0,
                'enhanced_brightness': 0,
                'boost_factor': 1.0,
                'process_fps': self.display_fps,
                'display_fps': self.display_fps,
                'mode': 'Auto' if hasattr(self.enhancer, 'auto_mode') and self.enhancer.auto_mode else 'Manual',
                'processing': 'GPU' if hasattr(self.enhancer, 'use_gpu') and self.enhancer.use_gpu else 'CPU'
            }

            # Add any missing keys to processing_stats
            for key, value in required_stats.items():
                if key not in processing_stats:
                    processing_stats[key] = value

            # If enhancer has overlay creation method, use it
            if self.enhancer is not None:
                try:
                    final_frame = self.enhancer.create_overlay(detected_frame, processing_stats)
                except Exception as e:
                    print(f"Error creating overlay: {e}")
                    # Fall back to basic overlay
                    cv2.putText(
                        final_frame,
                        f"Frame: {self.frame_count} | FPS: {self.display_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # in ms

        # Update frame counter
        self.frame_count += 1

        # Display additional info in console occasionally
        if self.frame_count % 30 == 0:
            print(f"Frame {self.frame_count} | Processing time: {process_time:.1f}ms")

        return final_frame

    def play(self):
        """Start video playback and processing"""
        if self.cap is None:
            print("No video loaded. Please load a video first.")
            return

        print("Starting video playback. Press 'q' to quit.")
        self.playing = True

        while self.playing:
            # Calculate actual FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                time_diff = current_time - self.last_frame_time
                if time_diff > 0:
                    self.display_fps = 1.0 / time_diff
            self.last_frame_time = current_time

            # Read the next frame
            ret, frame = self.cap.read()

            # If frame reading was not successful, we've reached the end of the video
            if not ret:
                print("End of video reached.")
                break

            # Store the current frame
            self.current_frame = frame

            # Process the frame
            self.processed_frame = self.process_frame(frame)

            # Display the processed frame
            cv2.imshow('Dark Vision Processor', self.processed_frame)

            # Wait for a key press (1ms) and handle it
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break

        # Release resources
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        print("Playback stopped.")

        # Print performance summary if enhancer is available
        if self.enhancer is not None:
            performance = self.enhancer.get_performance_summary()
            print("\nPerformance Summary:")
            print(f"Frames Processed: {performance.get('frames_processed', 0)}")
            print(f"Average Processing Time: {performance.get('avg_process_time_ms', 0):.2f} ms")
            print(f"Theoretical Max FPS: {performance.get('theoretical_max_fps', 0):.2f}")

    def _handle_key(self, key):
        """Handle keyboard input during playback"""
        if key == ord('q'):
            # Quit
            self.playing = False
            return False

        elif key == ord(' '):
            # Pause/Resume
            self.playing = not self.playing
            print(f"{'Paused' if not self.playing else 'Resumed'}")

            # If paused, wait for another space press to resume
            while not self.playing and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):
                    self.playing = True
                    print("Resumed")
                elif key == ord('q'):
                    return False
                else:
                    self._handle_key(key)

        elif key == ord('e'):
            # Toggle enhancer
            self.toggle_enhancer()

        elif key == ord('d'):
            # Toggle detector
            self.toggle_detector()

        elif key == ord('w'):
            # Toggle wireframe
            self.toggle_wireframe()

        elif key == ord('b'):
            # Toggle bounding boxes
            self.toggle_boxes()

        elif key == ord('o'):
            # Toggle overlay
            self.toggle_overlay()

        elif key == ord('g'):
            # Toggle GPU
            self.toggle_gpu()

        return True
