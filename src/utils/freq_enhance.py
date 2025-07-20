"""
src/utils/freq_enhance.py

Laplacian Pyramid based frequency enhancement algorithm for image enhancement.

This program implements a frequency-based approach to enhance low-light images using Laplacian pyramid decomposition.
Low-light images suffer from different problems at different frequency bands:

    1. Low-frequency component: overall illumination and contrast information.
    2. High-frequency component: details, edges, and noise.

By decomposing an image into these frequency bands using a Laplacian pyramid, we can:

    1. Selectively enhance the low-frequency components to improve global illumination and contrast.
    2. Process the high-frequency components to preserve details while reducing noise.

Functionality:
1. Decomposition: The input image is broken down into multiple frequency bands.
    - One low-frequency component; progressively smaller, more blurred versions of the image (Guassian pyramid)
    - Multiple high-frequency components; differences between agjacent levels of the Guassian pyramid (Laplacian pyramid)

2. Processing: Each frequency band can be independently enhanced:
    - Low-frequency component: exposure and contrast enhancement.
    - High-frequency component: detail preservation and noise reduction.

3. Reconstruction: The frequency bands are recombined to create the enhanced image.

This serves to reduce artifacts, preserve details, and improve overall image quality for the pose estimation model.

Strengths: Modest memory requirements, parallelism, GPU acceleration.
"""

import cv2
import numpy as np
import torch


class LaplacianPyramid:
    """
    Implements a Laplacian pyramid for frequency-based image enhancement.

    The Laplacian pyramid decomposes an image into frequency bands, allowing
    separate processing of global illumination and local details.
    """

    def __init__(self, num_levels=3):
        """
        Initialize the pyramid with the specified number of levels.

        Args:
            num_levels (int): Number of pyramid levels
        """
        self.num_levels = num_levels

    def decompose(self, image):
        """
        Decompose image into frequency components using Laplacian pyramid.

        Args:
            image: Input image as numpy array [H, W, C], [B, H, W, C] or
                   PyTorch tensor [C, H, W], [B, C, H, W]

        Returns:
            tuple: (low_freq, high_freqs)
                - low_freq: Low frequency component with shape [B, H, W, C] or [B, C, H, W]
                - high_freqs: List of high frequency components, each with shape [B, H, W, C] or [B, C, H, W]
        """
        # Determine if input is PyTorch tensor
        is_tensor = isinstance(image, torch.Tensor)
        original_device = None

        # Handle PyTorch tensor input
        if is_tensor:
            original_device = image.device

            # Save original shape for later
            if image.dim() == 3:  # [C, H, W]
                image = image.unsqueeze(0)  # Add batch dimension [1, C, H, W]

            # Convert tensor to numpy array for OpenCV operations
            image_np = image.detach().cpu().numpy()

            # Transpose from [B, C, H, W] to [B, H, W, C] for OpenCV
            image_np = image_np.transpose(0, 2, 3, 1)
        else:
            # Handle numpy array input
            if image.ndim == 3:  # [H, W, C]
                image_np = np.expand_dims(image, 0)  # Add batch dimension [1, H, W, C]
            else:
                image_np = image  # Already [B, H, W, C]

        # Get batch size
        batch_size = image_np.shape[0]

        # Initialize lists to store pyramid results for each batch item
        batch_low_freq = []
        batch_high_freqs = [[] for _ in range(self.num_levels)]

        # For each image in the batch
        for b in range(batch_size):
            img = image_np[b].copy()

            # Initialize fresh lists for this batch item
            gaussians = []
            laplacians = []

            # Build Gaussian pyramid
            current = img.copy()
            gaussians.append(current)

            for i in range(self.num_levels):
                current = cv2.pyrDown(current)
                gaussians.append(current)

            # Build Laplacian pyramid
            for i in range(self.num_levels):
                size = (gaussians[i].shape[1], gaussians[i].shape[0])
                expanded = cv2.pyrUp(gaussians[i + 1], dstsize=size)
                laplacian = cv2.subtract(gaussians[i], expanded)
                laplacians.append(laplacian)

            # The last level of Gaussian pyramid is the low-frequency component
            low_freq = gaussians[-1]
            batch_low_freq.append(low_freq)

            # Collect high-frequency components
            for i in range(self.num_levels):
                batch_high_freqs[i].append(laplacians[i])

        # Stack batch results
        low_freq_np = np.stack(batch_low_freq, axis=0)
        high_freqs_np = [np.stack(hf, axis=0) for hf in batch_high_freqs]

        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert from [B, H, W, C] to [B, C, H, W]
            low_freq_tensor = torch.from_numpy(low_freq_np.transpose(0, 3, 1, 2)).to(original_device)
            high_freqs_tensor = [torch.from_numpy(hf.transpose(0, 3, 1, 2)).to(original_device)
                                 for hf in high_freqs_np]
            return low_freq_tensor, high_freqs_tensor
        else:
            # Keep as numpy arrays
            return low_freq_np, high_freqs_np

    def reconstruct(self, low_freq, high_freq):
        """
        Reconstruct image from frequency components.

        Args:
            low_freq: Low frequency component
            high_freq: List of high frequency components

        Returns:
            Reconstructed image with same type as input
        """
        # Determine if input is PyTorch tensor
        is_tensor = isinstance(low_freq, torch.Tensor)
        original_device = None

        # Handle PyTorch tensor input
        if is_tensor:
            original_device = low_freq.device

            # Convert tensors to numpy arrays for OpenCV operations
            low_freq_np = low_freq.detach().cpu().numpy()
            high_freq_np = [hf.detach().cpu().numpy() for hf in high_freq]

            # Transpose from [B, C, H, W] to [B, H, W, C] for OpenCV
            low_freq_np = low_freq_np.transpose(0, 2, 3, 1)
            high_freq_np = [hf.transpose(0, 2, 3, 1) for hf in high_freq_np]
        else:
            # Already numpy arrays
            low_freq_np = low_freq
            high_freq_np = high_freq

        # Get batch size
        batch_size = low_freq_np.shape[0]

        # Initialize list to store reconstructed images
        reconstructed_batch = []

        # Process each image in the batch
        for b in range(batch_size):
            # Start with the low frequency component
            reconstructed = low_freq_np[b].copy()

            # Add high frequency components, from coarse to fine
            for i in range(len(high_freq_np) - 1, -1, -1):
                # Get the correct high frequency component for this level
                hf = high_freq_np[i][b]

                # Upscale reconstructed image to match high frequency component size
                if reconstructed.shape[:2] != hf.shape[:2]:
                    size = (hf.shape[1], hf.shape[0])  # width, height
                    reconstructed = cv2.pyrUp(reconstructed, dstsize=size)

                # Add high frequency component
                reconstructed = cv2.add(reconstructed, hf)

            reconstructed_batch.append(reconstructed)

        # Stack batch results
        reconstructed_np = np.stack(reconstructed_batch, axis=0)

        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert from [B, H, W, C] to [B, C, H, W]
            reconstructed_tensor = torch.from_numpy(reconstructed_np.transpose(0, 3, 1, 2)).to(original_device)
            return reconstructed_tensor
        else:
            # Keep as numpy array
            return reconstructed_np
