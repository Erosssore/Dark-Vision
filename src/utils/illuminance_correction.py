"""
src/utils/illuminance_correction.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminanceCorrection:
    """
    Dynamic illuminance correction with automatic device selection and batch processing.
    """

    def __init__(self, batch_size=16, force_cpu=False):
        """
        Initialize the illuminance correction module

        Args:
            batch_size: Size of batches for processing video frames
            force_cpu: Force CPU usage even if GPU is available
        """
        # Dynamic device selection
        self.device = self._select_device(force_cpu)
        print(f"Illuminance correction running on: {self.device}")

        self.batch_size = batch_size
        self.global_correction_network = self._build_global_correction_network().to(self.device)

    def _select_device(self, force_cpu=False):
        """
        Select the appropriate computation device

        Args:
            force_cpu: Force CPU usage even if GPU is available

        Returns:
            torch.device: The selected device
        """
        if force_cpu:
            return torch.device('cpu')

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Select first GPU by default
                device = torch.device('cuda:0')
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
                print(f"Using GPU: {gpu_name} with {vram:.2f} GB VRAM")
                return device

        # Default to CPU if no GPU is available or if there was an issue
        print("No GPU detected, using CPU")
        return torch.device('cpu')

    def _build_global_correction_network(self):
        """
        Build an efficient neural network to estimate global correction parameters
        """
        # Lightweight network for faster inference
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 3)  # 3 parameters for global correction: alpha, beta, gamma
        )

    def compute_local_correction_params(self, low_freq_component):
        """
        Compute spatially-varying correction parameters efficiently
        """
        # Using separable convolutions for efficiency
        kernel_size = 5
        padding = kernel_size // 2

        # Horizontal pass
        h_kernel = torch.ones(1, 1, 1, kernel_size) / kernel_size
        h_kernel = h_kernel.to(self.device)
        h_conv = F.conv2d(low_freq_component, h_kernel, padding=(0, padding), groups=1)

        # Vertical pass
        v_kernel = torch.ones(1, 1, kernel_size, 1) / kernel_size
        v_kernel = v_kernel.to(self.device)
        local_mean = F.conv2d(h_conv, v_kernel, padding=(padding, 0), groups=1)

        # Calculate correction factors
        local_params = torch.clamp(low_freq_component / (local_mean + 1e-6), 0.5, 2.0)

        return local_params

    def apply_taylor_series_correction(self, image, global_params, local_params):
        """
        Apply gamma correction using Taylor series approximation for efficiency
        """
        # Normalize image to [0, 1]
        normalized = torch.clamp(image, 0.0, 1.0)

        # Extract parameters
        alpha = global_params[:, 0].view(-1, 1, 1, 1)
        beta = global_params[:, 1].view(-1, 1, 1, 1)
        gamma = global_params[:, 2].view(-1, 1, 1, 1)

        # Apply correction (2-term Taylor approximation)
        shifted = normalized - 1.0
        term1 = 1.0 + gamma * shifted
        term2 = 0.5 * gamma * (gamma - 1.0) * (shifted * shifted)
        corrected = alpha * (term1 + term2) + beta

        # Apply local correction
        corrected = corrected * local_params

        return torch.clamp(corrected, 0.0, 1.0)

    def correct_batch(self, low_freq_batch):
        """
        Process a batch of low frequency components

        Args:
            low_freq_batch: Batch of low frequency components [B,H,W] or [B,1,H,W]

        Returns:
            Batch of corrected components
        """
        # Ensure input is a torch tensor with proper dimensions
        if not isinstance(low_freq_batch, torch.Tensor):
            low_freq_batch = torch.tensor(low_freq_batch, dtype=torch.float32)

        low_freq_batch = low_freq_batch.to(self.device)

        # Add channel dimension if needed
        if len(low_freq_batch.shape) == 3:  # [B,H,W]
            low_freq_batch = low_freq_batch.unsqueeze(1)  # [B,1,H,W]

        # 1. Global correction (using downsampled input for efficiency)
        downsampled = F.interpolate(
            low_freq_batch,
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        global_params = self.global_correction_network(downsampled)

        # 2. Local correction
        local_params = self.compute_local_correction_params(low_freq_batch)

        # 3. Apply correction
        corrected = self.apply_taylor_series_correction(
            low_freq_batch,
            global_params,
            local_params
        )

        return corrected

    def process_video(self, video_frames):
        """
        Process all frames from a video with batch processing

        Args:
            video_frames: List of video frames (low frequency components)

        Returns:
            List of corrected frames
        """
        results = []

        # Process in batches
        for i in range(0, len(video_frames), self.batch_size):
            # Get current batch
            batch = video_frames[i:i + self.batch_size]

            # Convert to tensors
            if isinstance(batch[0], np.ndarray):
                batch_tensor = torch.tensor(np.stack(batch), dtype=torch.float32)
            elif isinstance(batch[0], torch.Tensor):
                batch_tensor = torch.stack(batch)

            # Process batch
            with torch.no_grad():  # Disable gradient computation for inference
                corrected_batch = self.correct_batch(batch_tensor)

            # Convert back and append results
            for j in range(len(batch)):
                if j < corrected_batch.shape[0]:  # Safety check
                    frame = corrected_batch[j].squeeze().cpu().numpy()
                    results.append(frame)

        return results


# Function to determine optimal batch size based on GPU memory
def determine_optimal_batch_size(frame_height, frame_width):
    """
    Determine the optimal batch size based on available GPU memory

    Args:
        frame_height: Height of the video frames
        frame_width: Width of the video frames

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 4  # Conservative default for CPU

    # Get available GPU memory in bytes
    device = torch.cuda.current_device()
    available_memory = torch.cuda.get_device_properties(device).total_memory

    # Estimate memory needed per frame (in bytes)
    # Each pixel needs 4 bytes (float32)
    # We need memory for input, intermediate tensors, and output
    # Using a conservative estimate with 5x overhead
    memory_per_frame = frame_height * frame_width * 4 * 5

    # Calculate max batch size (use at most 70% of available memory)
    max_batch_size = int((available_memory * 0.7) / memory_per_frame)

    # Cap at reasonable values
    max_batch_size = max(1, min(64, max_batch_size))

    print(f"Automatically determined batch size: {max_batch_size}")
    return max_batch_size


# Utility function for single frame processing
def correct_illuminance(low_freq_component, force_cpu=False):
    """
    Apply illuminance correction to a single frame

    Args:
        low_freq_component: Low frequency component of the image
        force_cpu: Force CPU usage even if GPU is available

    Returns:
        Corrected component
    """
    # For single frame processing, use a batch size of 1
    corrector = IlluminanceCorrection(batch_size=1, force_cpu=force_cpu)

    # Handle single frame
    if isinstance(low_freq_component, np.ndarray):
        low_freq_component = torch.tensor(low_freq_component, dtype=torch.float32).unsqueeze(0)
    elif isinstance(low_freq_component, torch.Tensor) and len(low_freq_component.shape) == 2:
        low_freq_component = low_freq_component.unsqueeze(0)

    with torch.no_grad():
        result = corrector.correct_batch(low_freq_component)

    return result.squeeze().cpu().numpy() if isinstance(result, torch.Tensor) else result.squeeze()


# Singleton pattern with automatic re-creation if parameters change
_corrector_instance = None
_current_batch_size = None
_current_force_cpu = None


def get_corrector(batch_size=None, frame_height=None, frame_width=None, force_cpu=False):
    """
    Get or create a singleton corrector instance

    Args:
        batch_size: Size of batches for processing video frames (if None, will be determined automatically)
        frame_height: Height of the video frames (for automatic batch size determination)
        frame_width: Width of the video frames (for automatic batch size determination)
        force_cpu: Force CPU usage even if GPU is available

    Returns:
        IlluminanceCorrection instance
    """
    global _corrector_instance, _current_batch_size, _current_force_cpu

    # Determine batch size if not provided
    if batch_size is None and frame_height is not None and frame_width is not None:
        batch_size = determine_optimal_batch_size(frame_height, frame_width)
    elif batch_size is None:
        batch_size = 16  # Default if dimensions not provided

    # Check if we need to create a new instance
    if (_corrector_instance is None or
            _current_batch_size != batch_size or
            _current_force_cpu != force_cpu):
        # Create new instance with updated parameters
        _corrector_instance = IlluminanceCorrection(batch_size=batch_size, force_cpu=force_cpu)
        _current_batch_size = batch_size
        _current_force_cpu = force_cpu

    return _corrector_instance
