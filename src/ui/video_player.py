import cv2
import time
import os
import tkinter as tk
from tkinter import filedialog

class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.enhancer = None
        self.yolo_detector = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.playing = False
        self.current_frame = None
        self.processed_frame = None
        self.overlay_enabled = True
        self.enhancer_enabled = True
        self.detector_enabled = True
        self.last_frame_time = 0
        self.display_fps = 0
        self.loaded_video_path = None
        print("Video Player initialized")

    def set_yolo_detector(self, detector):
        self.yolo_detector = detector
        self.detector_enabled = detector is not None
        print(f"YOLO detector {'attached' if detector is not None else 'disabled'}")

    def set_enhancer(self, enhancer):
        self.enhancer = enhancer
        self.enhancer_enabled = enhancer is not None
        print(f"Video enhancer {'attached' if enhancer is not None else 'disabled'}")

    def toggle_enhancer(self):
        if self.enhancer is not None:
            self.enhancer_enabled = not self.enhancer_enabled
            print(f"Video enhancement: {'ON' if self.enhancer_enabled else 'OFF'}")
        else:
            print("No video enhancer available")

    def toggle_detector(self):
        if self.yolo_detector is not None:
            self.detector_enabled = not self.detector_enabled
            print(f"Human detection: {'ON' if self.detector_enabled else 'OFF'}")
        else:
            print("No detector available")

    def toggle_wireframe(self):
        if self.yolo_detector is not None:
            self.yolo_detector.toggle_wireframe()
            print(f"Wireframe visualization: {'ON' if self.yolo_detector.show_wireframe else 'OFF'}")
        else:
            print("No detector available")

    def toggle_boxes(self):
        if self.yolo_detector is not None:
            self.yolo_detector.toggle_boxes()
            print(f"Bounding boxes: {'ON' if self.yolo_detector.show_boxes else 'OFF'}")
        else:
            print("No detector available")

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        print(f"Information overlay: {'ON' if self.overlay_enabled else 'OFF'}")

    def toggle_gpu(self):
        if self.enhancer is not None:
            self.enhancer.toggle_gpu()
        if self.yolo_detector is not None:
            self.yolo_detector.toggle_gpu()

    def get_processed_video_path(self, original_path):
        base = os.path.splitext(os.path.basename(original_path))[0]
        processed_dir = "processed_videos"
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, f"{base}_processed.avi")
        return processed_path

    def load_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            self.loaded_video_path = video_path
            if not self.cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return False

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = 0

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
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if video_path:
            return self.load_video(video_path)
        else:
            print("No file selected")
            return False

    def process_frame(self, frame):
        if frame is None:
            return None
        processing_stats = {}
        enhanced_frame = frame.copy()
        if self.enhancer is not None and self.enhancer_enabled:
            try:
                enhanced_frame, enhancer_stats = self.enhancer.process(frame)
                processing_stats.update(enhancer_stats)
            except Exception as e:
                print(f"Error in enhancement processing: {e}")
                enhanced_frame = frame.copy()
        detected_frame = enhanced_frame.copy()
        if self.yolo_detector is not None and self.detector_enabled:
            try:
                detected_frame, detector_stats = self.yolo_detector.process(enhanced_frame)
                processing_stats.update(detector_stats)
            except Exception as e:
                print(f"Error in detection processing: {e}")
                detected_frame = enhanced_frame
        final_frame = detected_frame
        if self.overlay_enabled:
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
            for key, value in required_stats.items():
                if key not in processing_stats:
                    processing_stats[key] = value
            if self.enhancer is not None:
                try:
                    final_frame = self.enhancer.create_overlay(detected_frame, processing_stats)
                except Exception as e:
                    print(f"Error creating overlay: {e}")
                    cv2.putText(
                        final_frame,
                        f"Frame: {self.frame_count} | FPS: {self.display_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
        self.frame_count += 1
        return final_frame

    def play(self):
        if self.cap is None:
            print("No video loaded. Please load a video first.")
            return

        print("Starting video playback. Press 'q' to quit.")
        self.playing = True

        # Autosave processed video as AVI and MP4
        processed_avi_path = self.get_processed_video_path(self.loaded_video_path)
        processed_mp4_path = processed_avi_path.replace(".avi", ".mp4")
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')

        output_fps = 30  # <---- Force output FPS to 30

        out_avi = cv2.VideoWriter(
            processed_avi_path,
            fourcc_avi,
            output_fps,
            (self.width, self.height)
        )
        out_mp4 = cv2.VideoWriter(
            processed_mp4_path,
            fourcc_mp4,
            output_fps,
            (self.width, self.height)
        )
        print(f"Processed video will be autosaved to: {processed_avi_path} (AVI) and {processed_mp4_path} (MP4)")
        while self.playing:
            current_time = time.time()
            if self.last_frame_time > 0:
                time_diff = current_time - self.last_frame_time
                if time_diff > 0:
                    self.display_fps = 1.0 / time_diff
            self.last_frame_time = current_time

            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached.")
                break

            self.current_frame = frame
            self.processed_frame = self.process_frame(frame)

            cv2.imshow('Dark Vision Processor', self.processed_frame)
            out_avi.write(self.processed_frame)
            out_mp4.write(self.processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break

        if self.cap is not None:
            self.cap.release()
        out_avi.release()
        out_mp4.release()
        print("Processed video autosaved.")
        print(f"Saved AVI to: {processed_avi_path}")
        print(f"Saved MP4 to: {processed_mp4_path}")

        cv2.destroyAllWindows()
        print("Playback stopped.")

        if self.enhancer is not None:
            performance = self.enhancer.get_performance_summary()
            print("\nPerformance Summary:")
            print(f"Frames Processed: {performance.get('frames_processed', 0)}")
            print(f"Average Processing Time: {performance.get('avg_process_time_ms', 0):.2f} ms")
            print(f"Theoretical Max FPS: {performance.get('theoretical_max_fps', 0):.2f}")

    def _handle_key(self, key):
        if key == ord('q'):
            self.playing = False
            return False
        elif key == ord(' '):
            self.playing = not self.playing
            print(f"{'Paused' if not self.playing else 'Resumed'}")
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
            self.toggle_enhancer()
        elif key == ord('d'):
            self.toggle_detector()
        elif key == ord('w'):
            self.toggle_wireframe()
        elif key == ord('b'):
            self.toggle_boxes()
        elif key == ord('o'):
            self.toggle_overlay()
        elif key == ord('g'):
            self.toggle_gpu()
        return True