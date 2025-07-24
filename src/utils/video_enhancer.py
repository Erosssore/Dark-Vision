import cv2
import numpy as np
import time
from src.utils.freq_enhance import LaplacianPyramid


class VideoEnhancer:
    def __init__(self):
        # Configuration parameters
        self.num_levels = 2  # Fixed pyramid levels for Laplacian decomposition
        self.min_boost = 1.2  # Minimum brightness boost
        self.max_boost = 3.0  # Maximum brightness boost
        self.target_brightness = 120  # Target average brightness (0-255)
        self.manual_boost = 0.0  # Additional manual boost that can be adjusted
        self.auto_mode = True  # Start in auto mode

        # Performance tracking
        self.frame_count = 0
        self.total_process_time = 0
        self.fps_process = 0
        self.fps_display = 0
        self.frame_count_fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # Update FPS every half second

        # Initialize Laplacian Pyramid
        self.lap_pyramid = LaplacianPyramid(num_levels=self.num_levels)
        print(f"Initialized Laplacian Pyramid with {self.num_levels} levels")

        # Initialize OpenCL if available
        self.use_gpu = self._init_opencl()

    def _init_opencl(self):
        """Initialize OpenCL for GPU processing if available"""
        use_gpu = False
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                device = cv2.ocl.Device_getDefault()
                if device is not None:
                    vendor = device.vendorName()
                    print(f"OpenCL is available. Using device: {device.name()} from {vendor}")
                    use_gpu = True
                    return use_gpu
        except Exception as e:
            print(f"Error checking OpenCL support: {e}")

        print("OpenCL not available or failed to initialize. Falling back to CPU processing.")
        cv2.ocl.setUseOpenCL(False)
        return use_gpu

    def set_auto_mode(self, enabled):
        """Set auto brightness mode"""
        self.auto_mode = enabled
        print(f"Mode: {'Auto' if self.auto_mode else 'Manual'}")

    def set_boost_factor(self, adjustment):
        """Set manual brightness boost adjustment"""
        self.manual_boost = max(-1.0, min(3.0, adjustment))
        print(f"Manual brightness adjustment: {self.manual_boost:.1f}")

    def set_target_brightness(self, target):
        """Set target brightness for auto mode"""
        self.target_brightness = max(50, min(200, target))
        print(f"Target brightness: {self.target_brightness}")

    def toggle_gpu(self):
        """Toggle between GPU and CPU processing"""
        if cv2.ocl.haveOpenCL():
            self.use_gpu = not self.use_gpu
            cv2.ocl.setUseOpenCL(self.use_gpu)
            print(f"Processing mode: {'OpenCL (GPU)' if self.use_gpu else 'CPU'}")
        else:
            print("OpenCL not available")

    def process(self, frame):
        """
        Process a frame using adaptive frequency-based enhancement with
        improved handling for dark scenes and preservation of details.
        Returns: (enhanced_frame, stats_dict)
        """
        if frame is None:
            return None, {}

        # Make sure dimensions are even for pyramid operations
        height, width = frame.shape[:2]
        if height % 2 == 1 or width % 2 == 1:
            frame = cv2.resize(frame, (width - (width % 2), height - (height % 2)))

        # Process time measurement
        start_time = time.time()

        # Calculate current frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)

        # NEW: Calculate brightness distribution for more intelligent enhancement
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_pixels_ratio = np.sum(hist[:50]) / np.sum(hist)  # % of very dark pixels (0-50)
        bright_pixels_ratio = np.sum(hist[200:]) / np.sum(hist)  # % of very bright pixels (200-255)

        # NEW: Adaptive target brightness based on scene content
        # Lower target for scenes with lots of bright areas to avoid over-brightening
        adaptive_target = self.target_brightness
        if bright_pixels_ratio > 0.2:  # If more than 20% of pixels are bright
            adaptive_target = max(80, int(self.target_brightness * 0.8))  # Reduce target by 20%

        # Determine adaptive brightness boost factor with improved algorithm
        if self.auto_mode:
            # NEW: More conservative boosting that's stronger in darker areas
            if current_brightness < 40:  # Very dark scene
                # Higher boost for very dark scenes, but more controlled
                brightness_ratio = max(0.2, current_brightness / adaptive_target)
                adaptive_boost = self.min_boost + (self.max_boost - self.min_boost) * (1 - brightness_ratio ** 0.7)
            else:
                # Standard boost with logarithmic scaling for better mid-tone handling
                brightness_ratio = min(1.0, current_brightness / adaptive_target)
                # Use square root for more gradual boost reduction as brightness increases
                adaptive_boost = self.min_boost + (self.max_boost - self.min_boost) * (1 - brightness_ratio ** 0.5)

            # Apply user manual adjustment
            boost_factor = adaptive_boost + self.manual_boost
        else:
            boost_factor = self.min_boost + self.manual_boost

        # NEW: Cap boost factor to prevent extreme brightening
        boost_factor = min(boost_factor, 3.5)

        # Apply enhancement based on available processing method
        frame_float = frame.astype(np.float32)

        try:
            if self.use_gpu:
                # GPU processing with OpenCL
                frame_umat = cv2.UMat(frame_float)

                try:
                    # NEW: Improved OpenCL implementation with better frequency separation
                    # Create a Gaussian pyramid for better detail preservation
                    low_freq_umat = cv2.GaussianBlur(frame_umat, (9, 9), 2.0)
                    high_freq_umat = cv2.subtract(frame_umat, low_freq_umat)

                    # NEW: Apply adaptive brightening to low frequencies only
                    boosted_low_umat = cv2.multiply(low_freq_umat, float(boost_factor))

                    # NEW: Apply stronger sharpening to the high frequencies for better detail
                    sharpened_high = cv2.multiply(high_freq_umat, 1.2)  # Increase high-frequency contrast

                    # Combine the boosted low frequencies with enhanced high frequencies
                    enhanced_umat = cv2.add(boosted_low_umat, sharpened_high)
                    enhanced = cv2.convertScaleAbs(enhanced_umat).get()

                    # NEW: Apply local contrast enhancement if the scene is very dark
                    if dark_pixels_ratio > 0.4:  # If more than 40% of pixels are very dark
                        # Create CLAHE (Contrast Limited Adaptive Histogram Equalization)
                        enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_lab[:, :, 0] = clahe.apply(enhanced_lab[:, :, 0])
                        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                except Exception as ocl_error:
                    print(f"OpenCL error: {ocl_error}. Falling back to CPU for this frame.")
                    self.use_gpu = False  # Disable GPU for future frames
                    # Fallback to CPU processing with improved Laplacian Pyramid
                    low_freq, high_freq = self.lap_pyramid.decompose(frame_float)
                    enhanced_low_freq = low_freq * float(boost_factor)
                    # NEW: Enhance high frequencies for better detail preservation
                    enhanced_high_freq = high_freq * 1.2
                    enhanced = self.lap_pyramid.reconstruct(enhanced_low_freq, enhanced_high_freq)
                    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            else:
                # CPU processing with improved Laplacian Pyramid
                low_freq, high_freq = self.lap_pyramid.decompose(frame_float)

                # NEW: Apply adaptive brightening only to dark regions in low frequencies
                if dark_pixels_ratio > 0.3:  # If scene has significant dark areas
                    # Create a darkness mask to apply boost mainly to darker regions
                    gray_float = gray.astype(np.float32)
                    darkness_mask = 1.0 - np.clip(gray_float / 100.0, 0, 1)  # Stronger in darker areas

                    # Apply mask to low frequencies for brightness boost
                    for c in range(3):  # For each color channel
                        enhanced_low_freq_c = low_freq[:, :, c] * (1.0 + (boost_factor - 1.0) * darkness_mask)
                        low_freq[:, :, c] = enhanced_low_freq_c
                else:
                    # Standard boost for scenes without many dark areas
                    enhanced_low_freq = low_freq * float(boost_factor)
                    low_freq = enhanced_low_freq

                # NEW: Enhance high frequencies for better detail preservation and edge detection
                enhanced_high_freq = high_freq * 1.3  # Stronger enhancement of high frequencies

                # Reconstruct the image with enhanced frequencies
                enhanced = self.lap_pyramid.reconstruct(low_freq, enhanced_high_freq)

                # Ensure the enhanced frame is properly formatted
                if enhanced.ndim == 4:  # Handle batch dimension if present [B, H, W, C]
                    enhanced = enhanced[0]

                # NEW: Apply additional sharpening for better edge detection
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel * 0.5)  # Use 50% of the sharpening effect

                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"Error in enhancement processing: {e}")
            import traceback
            traceback.print_exc()
            # Return original frame if processing fails
            return frame, {"error": str(e)}

        # Calculate enhanced frame brightness for comparison
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        enhanced_brightness = np.mean(enhanced_gray)

        # NEW: Add a check to ensure we haven't over-brightened
        max_desired_brightness = min(180, int(self.target_brightness * 1.3))
        if enhanced_brightness > max_desired_brightness:
            # If too bright, apply a global dimming
            scale_factor = max_desired_brightness / enhanced_brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=scale_factor, beta=0)
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            enhanced_brightness = np.mean(enhanced_gray)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # ms
        self.total_process_time += process_time
        self.frame_count += 1
        self.frame_count_fps += 1

        # Update FPS values every interval
        current_time = time.time()
        if current_time - self.last_fps_update > self.fps_update_interval:
            time_diff = current_time - self.last_fps_update
            self.fps_process = 1000.0 / (self.total_process_time / self.frame_count) if self.frame_count > 0 else 0
            self.fps_display = self.frame_count_fps / time_diff
            self.frame_count_fps = 0
            self.last_fps_update = current_time

        # Return the enhanced frame and stats
        stats = {
            "frame_count": self.frame_count,
            "original_brightness": current_brightness,
            "enhanced_brightness": enhanced_brightness,
            "boost_factor": boost_factor,
            "dark_pixels_ratio": dark_pixels_ratio,
            "bright_pixels_ratio": bright_pixels_ratio,
            "process_fps": self.fps_process,
            "display_fps": self.fps_display,
            "mode": "Auto" if self.auto_mode else "Manual",
            "processing": "OpenCL (GPU)" if self.use_gpu else "CPU",
            "process_time_ms": process_time
        }

        return enhanced, stats

    def create_overlay(self, enhanced_frame, stats):
        """Create an informational overlay on the enhanced frame, including wireframe stats if present."""
        if enhanced_frame is None or stats is None:
            return enhanced_frame

        enhanced_copy = enhanced_frame.copy()

        # Font settings for display (smaller font)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1
        font_color = (255, 255, 255)

        # Prepare enhancement information text
        info_text = [
            f"Frame: {stats.get('frame_count', 0)}",
            f"Original Brightness: {stats.get('original_brightness', 0):.1f}/255",
            f"Enhanced Brightness: {stats.get('enhanced_brightness', 0):.1f}/255",
            f"Boost: {stats.get('boost_factor', 0):.2f}x",
            f"Process FPS: {stats.get('process_fps', 0):.1f}",
            f"Display FPS: {stats.get('display_fps', 0):.1f}",
            f"Mode: {stats.get('mode', '')}",
            f"Processing: {stats.get('processing', '')}",
        ]

        # Prepare wireframe information text if available
        wireframe_text = []
        if "humans_detected" in stats:
            wireframe_text.append(f"Humans detected: {stats['humans_detected']}")
        if "boxes" in stats and isinstance(stats["boxes"], list):
            wireframe_text.append(f"Boxes: {len(stats['boxes'])}")
        if "process_time" in stats:
            wireframe_text.append(f"Wireframe time: {stats['process_time']*1000:.1f} ms")
        if "process_time_ms" in stats:
            wireframe_text.append(f"Enhancer time: {stats['process_time_ms']:.1f} ms")

        # Combine all text for overlay
        all_text = info_text + wireframe_text

        # Calculate the size of the overlay dynamically
        line_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in all_text]
        max_width = max([size[0] for size in line_sizes]) + 20
        line_height = max([size[1] for size in line_sizes]) + 8
        overlay_height = len(all_text) * line_height + 20

        # Draw semi-transparent box
        overlay = enhanced_copy.copy()
        box_x1, box_y1 = 10, 10
        box_x2, box_y2 = box_x1 + max_width, box_y1 + overlay_height
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, enhanced_copy, 0.3, 0, enhanced_copy)

        # Draw text inside the box
        for i, text in enumerate(all_text):
            y_pos = box_y1 + line_height * (i + 1)
            cv2.putText(
                enhanced_copy,
                text,
                (box_x1 + 8, y_pos),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA
            )

        # Display processing info in console occasionally
        if stats.get('frame_count', 0) % 10 == 0:
            print(" | ".join(all_text))

        return enhanced_copy

    def get_performance_summary(self):
        """Get a summary of the enhancer's performance"""
        if self.frame_count > 0:
            avg_time = self.total_process_time / self.frame_count
            max_fps = 1000 / avg_time
            return {
                "frames_processed": self.frame_count,
                "avg_process_time_ms": avg_time,
                "theoretical_max_fps": max_fps,
                "current_fps": self.fps_process
            }
        return {"frames_processed": 0}