import cv2
import numpy as np
import time
from src.utils.freq_enhance import LaplacianPyramid


def apply_brightness_enhancement_opencl():
    # Check if OpenCL is available
    use_gpu = False
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            device = cv2.ocl.Device_getDefault()
            if device is not None:
                vendor = device.vendorName()
                print(f"OpenCL is available. Using device: {device.name()} from {vendor}")
                use_gpu = True
    except Exception as e:
        print(f"Error checking OpenCL support: {e}")
        print("Falling back to CPU processing.")
        cv2.ocl.setUseOpenCL(False)

    # Set video path directly in the script
    video_path = "C:/Users/Sam/Dark-Vision-Tests/DarkVideos/fnaf-gameplay.mp4"
    num_levels = 3  # Fixed pyramid levels

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video loaded: {width}x{height} at {fps} FPS")

    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Enhanced', cv2.WINDOW_NORMAL)

    # Initialize Laplacian Pyramid
    lap_pyramid = LaplacianPyramid(num_levels=num_levels)
    print(f"Initialized Laplacian Pyramid with {num_levels} levels")

    # Brightness enhancement parameters
    min_boost = 1.2  # Minimum brightness boost
    max_boost = 3.0  # Maximum brightness boost
    target_brightness = 120  # Target average brightness (0-255)

    # Manual adjustment
    manual_boost = 0.0  # Additional manual boost that can be adjusted
    auto_mode = True  # Start in auto mode

    # Process frames
    frame_count = 0
    total_time = 0

    # FPS calculation variables
    fps_process = 0
    fps_display = 0
    prev_time = time.time()
    fps_update_interval = 0.5  # Update FPS every half second
    last_fps_update = prev_time
    frame_count_fps = 0

    # Font settings for display
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.2
    font_thickness = 2
    font_color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # Make sure dimensions are even for pyramid operations
        if frame.shape[0] % 2 == 1 or frame.shape[1] % 2 == 1:
            frame = cv2.resize(frame, (width - (width % 2), height - (height % 2)))

        # Process time measurement
        start_time = time.time()

        try:
            # Calculate current frame brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)

            # Determine adaptive brightness boost factor
            if auto_mode:
                brightness_ratio = min(1.0, current_brightness / target_brightness)
                adaptive_boost = min_boost + (max_boost - min_boost) * (1 - brightness_ratio)
                boost_factor = adaptive_boost + manual_boost
            else:
                boost_factor = min_boost + manual_boost

            # Apply enhancement based on available processing method
            frame_float = frame.astype(np.float32)

            if use_gpu:
                try:
                    # GPU processing with OpenCL
                    frame_umat = cv2.UMat(frame_float)
                    low_freq_umat = cv2.GaussianBlur(frame_umat, (9, 9), 2.0)
                    high_freq_umat = cv2.subtract(frame_umat, low_freq_umat)
                    boosted_low_umat = cv2.multiply(low_freq_umat, float(boost_factor))
                    enhanced_umat = cv2.add(boosted_low_umat, high_freq_umat)
                    enhanced_umat = cv2.convertScaleAbs(enhanced_umat)
                    enhanced = enhanced_umat.get()
                except Exception as ocl_error:
                    print(f"OpenCL error: {ocl_error}. Falling back to CPU for this frame.")
                    # Fallback to CPU processing
                    low_freq, high_freq = lap_pyramid.decompose(frame_float)
                    enhanced_low_freq = low_freq * float(boost_factor)
                    enhanced = lap_pyramid.reconstruct(enhanced_low_freq, high_freq)
                    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            else:
                # CPU processing with Laplacian Pyramid
                low_freq, high_freq = lap_pyramid.decompose(frame_float)
                enhanced_low_freq = low_freq * float(boost_factor)
                enhanced = lap_pyramid.reconstruct(enhanced_low_freq, high_freq)

                # Ensure the enhanced frame is properly formatted
                if enhanced.ndim == 4:  # Handle batch dimension if present [B, H, W, C]
                    enhanced = enhanced[0]
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

            # Calculate enhanced frame brightness for comparison
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            enhanced_brightness = np.mean(enhanced_gray)

            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # ms
            total_time += process_time

            # Update FPS calculations
            current_time = time.time()
            frame_count += 1
            frame_count_fps += 1

            # Update FPS values every interval
            if current_time - last_fps_update > fps_update_interval:
                time_diff = current_time - last_fps_update
                fps_process = 1000.0 / (total_time / frame_count) if frame_count > 0 else 0
                fps_display = frame_count_fps / time_diff
                frame_count_fps = 0
                last_fps_update = current_time

            # Display info overlay on enhanced frame
            enhanced_copy = enhanced.copy()
            info_text = [
                f"Frame: {frame_count}",
                f"Original Brightness: {current_brightness:.1f}/255",
                f"Enhanced Brightness: {enhanced_brightness:.1f}/255",
                f"Boost: {boost_factor:.2f}x",
                f"Process FPS: {fps_process:.1f}",
                f"Display FPS: {fps_display:.1f}",
                f"Mode: {'Auto' if auto_mode else 'Manual'}",
                f"Processing: {'OpenCL (GPU)' if use_gpu else 'CPU'}"
            ]

            # Create a semi-transparent overlay for better text readability
            overlay = enhanced_copy.copy()
            line_height = int(30 * font_scale)
            overlay_height = len(info_text) * line_height + 20
            cv2.rectangle(overlay, (10, 10), (420, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, enhanced_copy, 0.3, 0, enhanced_copy)

            # Draw text with the sans-serif font
            for i, text in enumerate(info_text):
                y_pos = 30 + i * line_height
                cv2.putText(
                    enhanced_copy,
                    text,
                    (20, y_pos),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv2.LINE_AA
                )

            # Display processing info in console occasionally
            if frame_count % 10 == 0:
                print(f"Frame {frame_count} | Brightness: {current_brightness:.1f} → {enhanced_brightness:.1f} | "
                      f"Boost: {boost_factor:.2f}x | Process FPS: {fps_process:.1f} | Display FPS: {fps_display:.1f}")

            # Display the frames
            cv2.imshow('Original', frame)
            cv2.imshow('Enhanced', enhanced_copy)

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Increase manual brightness boost
            manual_boost += 0.2
            print(f"Manual brightness adjustment: +{manual_boost:.1f}")
        elif key == ord('z'):
            # Decrease manual brightness boost
            manual_boost = max(-1.0, manual_boost - 0.2)
            print(f"Manual brightness adjustment: {manual_boost:.1f}")
        elif key == ord('m'):
            # Toggle between auto and manual mode
            auto_mode = not auto_mode
            print(f"Mode: {'Auto' if auto_mode else 'Manual'}")
        elif key == ord('r'):
            # Reset manual adjustment
            manual_boost = 0.0
            print("Manual adjustment reset")
        elif key == ord('t'):
            # Increase target brightness
            target_brightness = min(200, target_brightness + 10)
            print(f"Target brightness: {target_brightness}")
        elif key == ord('g'):
            # Decrease target brightness
            target_brightness = max(50, target_brightness - 10)
            print(f"Target brightness: {target_brightness}")
        elif key == ord('p'):
            # Toggle GPU processing if available
            if cv2.ocl.haveOpenCL():
                use_gpu = not use_gpu
                cv2.ocl.setUseOpenCL(use_gpu)
                print(f"Processing mode: {'OpenCL (GPU)' if use_gpu else 'CPU'}")
            else:
                print("OpenCL not available")

    # Print summary
    if frame_count > 0:
        print(f"\nProcessed {frame_count} frames")
        print(f"Average processing time: {total_time / frame_count:.2f} ms")
        print(f"Approximate max FPS: {1000 / (total_time / frame_count):.1f}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    apply_brightness_enhancement_opencl()
