"""
src/tests/freq_enhance_test.py
"""

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.freq_enhance import LaplacianPyramid


class TestLaplacianPyramid(unittest.TestCase):
    """Test suite for the LaplacianPyramid class"""

    def setUp(self):
        """Setup tests environment"""
        # Create pyramid with different levels for testing
        self.pyramid_3 = LaplacianPyramid(num_levels=3)
        self.pyramid_5 = LaplacianPyramid(num_levels=5)

        # Create tests images
        # Single image - numpy
        self.np_img = np.random.rand(256, 256, 3).astype(np.float32)

        # Batch of images - numpy
        self.np_batch = np.random.rand(4, 256, 256, 3).astype(np.float32)

        # Single image - torch
        self.torch_img = torch.rand(3, 256, 256, dtype=torch.float32)

        # Batch of images - torch
        self.torch_batch = torch.rand(4, 3, 256, 256, dtype=torch.float32)

        # Make a tests image with clear frequency components
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xx, yy = np.meshgrid(x, y)

        # Low frequency component (smooth gradient)
        low_freq = xx * yy

        # Mid frequency component (medium waves)
        mid_freq = 0.2 * np.sin(xx * 10 * np.pi) * np.cos(yy * 10 * np.pi)

        # High frequency component (fine details)
        high_freq = 0.1 * np.sin(xx * 50 * np.pi) * np.cos(yy * 50 * np.pi)

        # Combine components
        combined = low_freq + mid_freq + high_freq
        combined = np.clip(combined, 0, 1)

        # Create RGB synthetic tests image
        self.synthetic_img = np.stack([combined, combined * 0.8, combined * 0.6], axis=-1).astype(np.float32)

        # Convert to torch
        self.synthetic_torch = torch.from_numpy(self.synthetic_img.transpose(2, 0, 1))

    def test_decompose_numpy_single(self):
        """Test decomposition with single numpy image"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.np_img)

        # Check types
        self.assertIsInstance(low_freq, np.ndarray)
        self.assertIsInstance(high_freqs, list)
        self.assertIsInstance(high_freqs[0], np.ndarray)

        # Check shapes
        self.assertEqual(low_freq.shape[0], 1)  # Batch dim should be 1
        self.assertEqual(len(high_freqs), 3)  # 3 levels

    def test_decompose_numpy_batch(self):
        """Test decomposition with batch of numpy images"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.np_batch)

        # Check batch size preservation
        self.assertEqual(low_freq.shape[0], 4)
        self.assertEqual(high_freqs[0].shape[0], 4)

    def test_decompose_torch_single(self):
        """Test decomposition with single torch tensor"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.torch_img)

        # Check types
        self.assertIsInstance(low_freq, torch.Tensor)
        self.assertIsInstance(high_freqs, list)
        self.assertIsInstance(high_freqs[0], torch.Tensor)

        # Check shapes and channel order
        self.assertEqual(low_freq.dim(), 4)  # [B, C, H, W]
        self.assertEqual(low_freq.shape[0], 1)  # Batch dim should be 1
        self.assertEqual(low_freq.shape[1], 3)  # Channel dim should be 3

    def test_decompose_torch_batch(self):
        """Test decomposition with batch of torch tensors"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.torch_batch)

        # Check batch size preservation
        self.assertEqual(low_freq.shape[0], 4)
        self.assertEqual(high_freqs[0].shape[0], 4)

    def test_reconstruct_numpy(self):
        """Test reconstruction with numpy arrays"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.np_img)
        reconstructed = self.pyramid_3.reconstruct(low_freq, high_freqs)

        # Check type
        self.assertIsInstance(reconstructed, np.ndarray)

        # Check shape
        self.assertEqual(reconstructed.shape, (1,) + self.np_img.shape)

        # Check reconstruction accuracy
        error = np.abs(reconstructed[0] - self.np_img).mean()
        self.assertLess(error, 0.01, f"Reconstruction error {error} is too high")

    def test_reconstruct_torch(self):
        """Test reconstruction with torch tensors"""
        low_freq, high_freqs = self.pyramid_3.decompose(self.torch_img)
        reconstructed = self.pyramid_3.reconstruct(low_freq, high_freqs)

        # Check type
        self.assertIsInstance(reconstructed, torch.Tensor)

        # Check shape
        self.assertEqual(reconstructed.shape, (1,) + self.torch_img.shape)

        # Check reconstruction accuracy
        error = torch.abs(reconstructed[0] - self.torch_img).mean().item()
        self.assertLess(error, 0.01, f"Reconstruction error {error} is too high")

    def test_batch_consistency(self):
        """Test consistency across batch items"""
        # Process single images one by one
        results_individual = []
        for i in range(4):
            single_img = self.torch_batch[i:i + 1]
            low_freq, high_freqs = self.pyramid_3.decompose(single_img)
            reconstructed = self.pyramid_3.reconstruct(low_freq, high_freqs)
            results_individual.append(reconstructed)

        # Concatenate individual results
        individual_batch = torch.cat(results_individual, dim=0)

        # Process the whole batch at once
        low_freq, high_freqs = self.pyramid_3.decompose(self.torch_batch)
        batch_result = self.pyramid_3.reconstruct(low_freq, high_freqs)

        # Compare results - should be identical
        error = torch.abs(individual_batch - batch_result).mean().item()
        self.assertLess(error, 1e-5, f"Batch processing inconsistency: {error}")

    def test_different_levels(self):
        """Test with different number of pyramid levels"""
        # Test with 3 levels
        low_freq_3, high_freqs_3 = self.pyramid_3.decompose(self.torch_img)
        reconstructed_3 = self.pyramid_3.reconstruct(low_freq_3, high_freqs_3)

        # Test with 5 levels
        low_freq_5, high_freqs_5 = self.pyramid_5.decompose(self.torch_img)
        reconstructed_5 = self.pyramid_5.reconstruct(low_freq_5, high_freqs_5)

        # Both should reconstruct the original image well
        error_3 = torch.abs(reconstructed_3[0] - self.torch_img).mean().item()
        error_5 = torch.abs(reconstructed_5[0] - self.torch_img).mean().item()

        self.assertLess(error_3, 0.01, f"3-level reconstruction error {error_3} is too high")
        self.assertLess(error_5, 0.01, f"5-level reconstruction error {error_5} is too high")

    def test_synthetic_image_decomposition(self):
        """Test decomposition quality with synthetic image containing known frequency components"""
        # Decompose the synthetic image
        low_freq, high_freqs = self.pyramid_3.decompose(self.synthetic_img)

        # The synthetic image has clear frequency separation,
        # so the pyramid levels should capture different frequency bands

        # Low frequency should capture the gradient
        low_var = np.var(low_freq[0])

        # High frequency should capture the fine details
        high_var = np.var(high_freqs[-1][0])

        # The variance of high frequency component should be less than low frequency
        self.assertGreater(low_var, high_var)

        # Visual check (optional - disabled in automated testing)
        if False:  # Set to True for visual debugging
            self.visualize_pyramid(self.synthetic_img, low_freq, high_freqs)

    def test_device_preservation(self):
        """Test that tensor device is preserved"""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Move tensor to GPU
        gpu_tensor = self.torch_batch.cuda()

        # Process
        low_freq, high_freqs = self.pyramid_3.decompose(gpu_tensor)
        reconstructed = self.pyramid_3.reconstruct(low_freq, high_freqs)

        # Check device
        self.assertTrue(low_freq.is_cuda)
        self.assertTrue(all(hf.is_cuda for hf in high_freqs))
        self.assertTrue(reconstructed.is_cuda)

    def visualize_pyramid(self, image, low_freq, high_freqs):
        """Helper to visualize pyramid decomposition - not an actual tests"""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
            low_freq = low_freq.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            high_freqs = [hf.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] for hf in high_freqs]
        else:
            if image.ndim == 3:  # [H, W, C]
                pass
            else:  # [B, H, W, C]
                image = image[0]
            low_freq = low_freq[0]
            high_freqs = [hf[0] for hf in high_freqs]

        # Create figure
        fig, axes = plt.subplots(1, 2 + len(high_freqs), figsize=(15, 5))

        # Plot original image
        axes[0].imshow(np.clip(image, 0, 1))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot low frequency component
        axes[1].imshow(np.clip(low_freq, 0, 1))
        axes[1].set_title('Low Frequency')
        axes[1].axis('off')

        # Plot high frequency components
        for i, hf in enumerate(high_freqs):
            # Normalize high freq component for better visualization
            hf_vis = (hf - hf.min()) / (hf.max() - hf.min() + 1e-5)
            axes[i + 2].imshow(hf_vis)
            axes[i + 2].set_title(f'High Freq {i + 1}')
            axes[i + 2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
