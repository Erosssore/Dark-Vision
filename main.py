import argparse
from src.utils.video_enhancer import VideoEnhancer
from src.models.wireframe_detector import YOLOv8PoseDetector
from src.ui.video_player import VideoPlayer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Enhancement and Human Detection with YOLOv8")
    parser.add_argument("--video-path", help="Path to the video file (optional, will open file dialog if not provided)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable video enhancement")
    parser.add_argument("--no-detection", action="store_true", help="Disable human detection")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Human detection confidence threshold")
    parser.add_argument("--no-wireframe", action="store_true", help="Disable wireframe visualization")
    parser.add_argument("--no-boxes", action="store_true", help="Disable bounding boxes")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for YOLO processing if available")
    parser.add_argument("--yolo-model", type=str, default="yolov8n-pose.pt", help="YOLOv8 pose model path or name")
    parser.add_argument("--batch", action="store_true", help="Batch process video and save result (no live playback)")
    parser.add_argument("--output-path", type=str, default=None, help="Output video file path for batch processing (optional)")
    parser.add_argument("--denoise", action="store_true", help="Enable denoising by default")
    args = parser.parse_args()

    # Create video player
    player = VideoPlayer()

    # Load video from path or open file dialog
    video_loaded = False
    if args.video_path:
        video_loaded = player.load_video(args.video_path)

    # If no video path provided or loading failed, open file dialog
    if not args.video_path or not video_loaded:
        video_loaded = player.select_video_file()

    # Exit if no video was loaded
    if not video_loaded:
        print("No video loaded. Exiting.")
        return

    # Create and attach video enhancer
    enhancer = VideoEnhancer()
    player.set_enhancer(enhancer)
    if args.denoise and hasattr(enhancer, "toggle_denoise"):
        enhancer.toggle_denoise()

    # --- Attach YOLOv8PoseDetector ---
    detector = YOLOv8PoseDetector(
        confidence_threshold=args.confidence,
        model_path=args.yolo_model
    )
    if args.no_wireframe and hasattr(detector, "toggle_wireframe"):
        detector.toggle_wireframe()
    if args.no_boxes and hasattr(detector, "toggle_boxes"):
        detector.toggle_boxes()
    if args.no_detection and hasattr(detector, "toggle"):
        detector.toggle()
    # GPU toggle if implemented in your detector
    if args.use_gpu and hasattr(detector, "toggle_gpu"):
        detector.toggle_gpu()
    player.set_yolo_detector(detector)
    # ----------------------------------

    # Apply other command line options
    if args.no_enhance:
        player.toggle_enhancer()

    # Batch processing mode
    if args.batch:
        # Determine output path
        output_path = args.output_path
        if output_path is None:
            output_path = player.get_processed_video_path(player.loaded_video_path)
        # Save processed video (no live playback)
        print(f"Batch processing video and saving to {output_path}...")
        # Make sure to reset to beginning
        player.cap.set(0, 0)
        # Save processed to output_path
        # Temporarily change processed path for batch, then restore
        orig_loaded_path = player.loaded_video_path
        player.loaded_video_path = output_path  # To save with correct name
        player.play(save_processed=True)
        player.loaded_video_path = orig_loaded_path  # Restore original after batch
        print(f"Batch processing done! Playing processed video...")

        # Load and play processed video file
        video_loaded = player.load_video(output_path)
        if video_loaded:
            player.play()
        else:
            print("Failed to load processed video for playback.")
        return

    # Display keyboard controls
    print("\nKeyboard Controls:")
    print("  q : Quit")
    print("  Space : Pause/Resume")
    print("  e : Toggle enhancer visuals")
    print("  d : Toggle human detection")
    print("  w : Toggle wireframe visualization")
    print("  b : Toggle bounding boxes")
    print("  o : Toggle overlay")
    print("  g : Toggle GPU acceleration")
    print("\n")

    # Start playback (live)
    player.play()

if __name__ == "__main__":
    main()