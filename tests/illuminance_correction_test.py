"""
src/tests/illuminance_correction_test.py

Test suite for the illuminance correction module.

This test module performs a comprehensive set of tests for the illuminance
correction functionality, including device selection, batch processing,
performance comparison, and real-world use cases.

Tests performed:

1. Device Selection (test_device_selection)
   - Tests automatic GPU/CPU detection
   - Tests forced CPU mode

2. Batch Size Determination (test_batch_size_determination)
   - Tests automatic determination of optimal batch size for different frame sizes
   - Verifies functionality with and without GPU

3. Single Frame Correction (test_single_frame_correction)
   - Tests correction of a single image using numpy arrays
   - Tests correction of a single image using torch tensors
   - Saves visualization comparing original and corrected images

4. Batch Processing (test_batch_processing)
   - Tests processing of multiple frames with batching
   - Verifies correct output shape and size
   - Measures and reports processing time

5. Performance Comparison (test_performance_comparison)
   - Compares performance across different configurations:
     - Different batch sizes (1, 4, 16)
     - Different devices (CPU vs GPU if available)
   - Generates performance visualization showing frames per second

6. Video File Processing (test_video_file_processing)
   - Tests processing of an actual video file
   - Creates a test video if none is available
   - Measures overall processing time and FPS

7. Real World Example (test_real_world_example)
   - Tests frequency decomposition pipeline:
     - Separation of image into low and high frequency components
     - Correction of low frequency component
     - Recombination of frequency components
   - Visualizes each step of the process

Usage:
    python -m unittest tests.test_illuminance_correction

Output:
    Test results will be saved to the tests/output directory, including:
    - Visualizations of original vs corrected images
    - Performance comparison charts
    - Processed test video
"""

import os
import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt
from pathlib import Path
import unittest
import sys

# Add the src directory to the Python path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module from src
from src.utils.illuminance_correction import (
    IlluminanceCorrection,
    correct_illuminance,
    get_corrector,
    determine_optimal_batch_size
)


class TestIlluminanceCorrection(unittest.TestCase):
    """
    Test suite for the illuminance correction module.
    """

    def setUp(self):
        """
        Set up test environment
        """
        # Create test directory if it doesn't exist
        self.test_output_dir = Path("illuminance_tests/output")
        self.test_output_dir.mkdir(exist_ok=True, parents=True)

        # Generate synthetic test data
        self.generate_test_data()

    def generate_test_data(self):
        """
        Generate synthetic test data for testing
        """
        # Create a synthetic low frequency component with uneven illumination
        h, w = 256, 256

        # Create gradient for uneven illumination
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Generate test image with non-uniform illumination
        # Bright in top-left, dark in bottom-right
        self.test_image = 0.8 * (1 - xx * yy) + 0.2

        # Create a sequence of images with varying illumination for video test
        self.test_video_frames = []
        for i in range(10):
            # Vary illumination across frames
            factor = 0.5 + 0.5 * np.sin(i * 0.6)
            frame = factor * self.test_image
            self.test_video_frames.append(frame)

    def test_device_selection(self):
        """
        Test automatic device selection
        """
        # Test with auto-detection
        corrector = IlluminanceCorrection(batch_size=4)
        print(f"Auto-detected device: {corrector.device}")

        # Test with forced CPU
        cpu_corrector = IlluminanceCorrection(batch_size=4, force_cpu=True)
        self.assertEqual(cpu_corrector.device, torch.device('cpu'))
        print("Device selection test passed")

    def test_batch_size_determination(self):
        """
        Test automatic batch size determination
        """
        # Test with different frame sizes
        sizes = [(256, 256), (512, 512), (1080, 1920)]

        for h, w in sizes:
            batch_size = determine_optimal_batch_size(h, w)
            print(f"Optimal batch size for {h}x{w}: {batch_size}")

        # Should still work even without GPU
        with_cpu = determine_optimal_batch_size(1080, 1920)
        self.assertGreater(with_cpu, 0)

    def test_single_frame_correction(self):
        """
        Test correction of a single frame
        """
        # Test numpy input
        result_numpy = correct_illuminance(self.test_image)
        self.assertEqual(result_numpy.shape, self.test_image.shape)

        # Test tensor input
        tensor_input = torch.tensor(self.test_image, dtype=torch.float32)
        result_tensor = correct_illuminance(tensor_input)
        self.assertEqual(result_tensor.shape, self.test_image.shape)

        # Save visualization
        self._save_comparison(
            self.test_image,
            result_numpy,
            "single_frame_correction.png"
        )

    def test_batch_processing(self):
        """
        Test batch processing of multiple frames
        """
        # Get corrector with batch size 4
        corrector = get_corrector(batch_size=4)

        # Process video frames
        start_time = time.time()
        corrected_frames = corrector.process_video(self.test_video_frames)
        processing_time = time.time() - start_time

        # Verify results
        self.assertEqual(len(corrected_frames), len(self.test_video_frames))

        # Save visualization of first and last frame
        self._save_comparison(
            self.test_video_frames[0],
            corrected_frames[0],
            "batch_first_frame.png"
        )
        self._save_comparison(
            self.test_video_frames[-1],
            corrected_frames[-1],
            "batch_last_frame.png"
        )

        print(f"Batch processing of {len(self.test_video_frames)} frames took {processing_time:.4f} seconds")

    def test_performance_comparison(self):
        """
        Compare performance of different configurations
        """
        # Test parameters
        batch_sizes = [1, 4, 16]
        devices = [False, True]  # False = use GPU if available, True = force CPU

        results = []

        for force_cpu in devices:
            device_name = "CPU" if force_cpu else "GPU"
            for batch_size in batch_sizes:
                # Skip larger batch sizes for CPU as they're very slow
                if force_cpu and batch_size > 4:
                    continue

                # Get corrector
                corrector = get_corrector(batch_size=batch_size, force_cpu=force_cpu)

                # Measure processing time
                start_time = time.time()
                _ = corrector.process_video(self.test_video_frames)
                elapsed = time.time() - start_time

                # Record result
                results.append({
                    "device": device_name,
                    "batch_size": batch_size,
                    "time": elapsed,
                    "frames_per_second": len(self.test_video_frames) / elapsed
                })

                print(f"{device_name} with batch size {batch_size}: {elapsed:.4f}s " +
                      f"({len(self.test_video_frames) / elapsed:.2f} FPS)")

        # Save performance results
        self._save_performance_plot(results)

    def test_video_file_processing(self):
        """
        Test processing an actual video file if available
        """
        # Check if test video exists
        test_video_path = "illuminance_tests/test_data/test_video.mp4"
        if not os.path.exists(test_video_path):
            # Create test data directory if needed
            test_data_dir = Path("illuminance_tests/test_data")
            test_data_dir.mkdir(exist_ok=True, parents=True)
            test_video_path = str(test_data_dir / "test_video.mp4")

            # Create a simple test video from synthetic data
            self._create_test_video(test_video_path)

        if os.path.exists(test_video_path):
            output_path = str(self.test_output_dir / "processed_video.mp4")
            self._process_test_video(test_video_path, output_path)

            # Verify output was created
            self.assertTrue(os.path.exists(output_path))
            print(f"Processed video saved to {output_path}")
        else:
            print("Skipping video file test, no test video available")

    def _save_comparison(self, original, corrected, filename):
        """
        Save a comparison visualization
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(corrected, cmap='gray')
        plt.title("Corrected")
        plt.axis('off')

        # Add difference visualization
        plt.subplot(1, 3, 3)
        diff = np.abs(corrected - original)
        plt.imshow(diff, cmap='hot')
        plt.title("Difference")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(self.test_output_dir / filename))
        plt.close()

    def _save_performance_plot(self, results):
        """
        Save performance comparison plot
        """
        plt.figure(figsize=(10, 6))

        # Group by device
        gpu_results = [r for r in results if r["device"] == "GPU"]
        cpu_results = [r for r in results if r["device"] == "CPU"]

        # Plot GPU results
        if gpu_results:
            plt.plot(
                [r["batch_size"] for r in gpu_results],
                [r["frames_per_second"] for r in gpu_results],
                'o-',
                label="GPU"
            )

        # Plot CPU results
        if cpu_results:
            plt.plot(
                [r["batch_size"] for r in cpu_results],
                [r["frames_per_second"] for r in cpu_results],
                's-',
                label="CPU"
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Frames Per Second")
        plt.title("Performance Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(str(self.test_output_dir / "performance_comparison.png"))
        plt.close()

    def _create_test_video(self, output_path):
        """
        Create a synthetic test video for testing
        """
        # Create more frames for a proper video
        frames = []
        num_frames = 30

        for i in range(num_frames):
            # Create frame with varying illumination
            angle = i * 2 * np.pi / num_frames
            factor = 0.5 + 0.4 * np.sin(angle)

            # Apply vignette effect
            h, w = 256, 256
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            dist = np.sqrt(xx ** 2 + yy ** 2)
            vignette = np.clip(1 - dist ** 2, 0.2, 1.0)

            # Create frame with illumination issues
            frame = factor * vignette * np.ones((h, w))

            # Add some content
            frame[h // 4:3 * h // 4, w // 4:3 * w // 4] += 0.2 * np.sin(dist[h // 4:3 * h // 4, w // 4:3 * w // 4] * 10)

            # Normalize to [0, 1]
            frame = np.clip(frame, 0, 1)

            # Convert to uint8 for video
            frame_uint8 = (frame * 255).astype(np.uint8)

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
            frames.append(frame_bgr)

        # Save as video
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release()
        print(f"Created test video at {output_path}")

    def _process_test_video(self, input_path, output_path):
        """
        Process a test video file
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get corrector with automatic batch size
        corrector = get_corrector(frame_height=height, frame_width=width)

        # Extract frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for simplicity in this test
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray / 255.0)  # Normalize to [0, 1]

        cap.release()

        # Process all frames
        start_time = time.time()
        corrected_frames = corrector.process_video(frames)
        processing_time = time.time() - start_time

        # Write output video
        for corrected in corrected_frames:
            # Convert back to uint8 BGR
            frame_uint8 = (corrected * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
            writer.write(frame_bgr)

        writer.release()

        print(f"Processed {len(frames)} frames in {processing_time:.2f}s " +
              f"({len(frames) / processing_time:.2f} FPS)")

    def test_real_world_example(self):
        """
        Test with a real-world example using frequency decomposition
        """
        try:
            # Skip if PIL not available
            from PIL import Image
            import numpy as np

            # Create synthetic image with details and uneven illumination
            h, w = 512, 512

            # Create illumination gradient
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            illumination = 0.7 * (1 - xx * yy) + 0.3

            # Create high-frequency details (simulated texture)
            noise = np.random.randn(h, w) * 0.05

            # Add text (high frequency)
            text_mask = np.zeros((h, w))
            text_width = 300
            text_height = 50
            text_y = h // 2 - text_height // 2
            text_x = w // 2 - text_width // 2
            text_mask[text_y:text_y + text_height, text_x:text_x + text_width] = 1

            # Combine to create test image
            image = illumination + noise * text_mask
            image = np.clip(image, 0, 1)

            # Decompose into frequency components (simulate your pipeline)
            # Here we're just using a simple blur for demonstration
            import cv2
            low_freq = cv2.GaussianBlur(image, (21, 21), 5)
            high_freq = image - low_freq

            # Correct low frequency component
            corrected_low_freq = correct_illuminance(low_freq)

            # Recombine
            corrected_image = corrected_low_freq + high_freq
            corrected_image = np.clip(corrected_image, 0, 1)

            # Save results
            self._save_comparison(image, corrected_image, "real_world_example.png")

            # Additional visualization: show frequency components
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(low_freq, cmap='gray')
            plt.title("Low Frequency Component")
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(high_freq, cmap='gray', vmin=-0.1, vmax=0.1)
            plt.title("High Frequency Component")
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.imshow(corrected_image, cmap='gray')
            plt.title("Corrected Image")
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(corrected_low_freq, cmap='gray')
            plt.title("Corrected Low Frequency")
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(corrected_image - image, cmap='hot')
            plt.title("Difference")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(str(self.test_output_dir / "frequency_components.png"))
            plt.close()

        except ImportError:
            print("Skipping real-world example test, PIL not available")


# Main execution
if __name__ == "__main__":
    unittest.main()
