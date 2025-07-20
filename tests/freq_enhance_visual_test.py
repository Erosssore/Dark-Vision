"""
src/tests/freq_enhance_visual_test.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import json
from src.utils.freq_enhance import LaplacianPyramid


def load_config(config_path="image_paths.json"):
    """Load image paths from a configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        # Return a default empty config
        return {"dark_images": {}}


def save_config(config, config_path="image_paths.json"):
    """Save image paths to a configuration file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Error saving config file: {e}")


def enhance_dark_image(image, levels=4, low_boost=1.5, high_boost=2.0):
    """
    Enhance a dark image using Laplacian pyramid frequency manipulation.

    Args:
        image: Input image (numpy array [H,W,C] or torch tensor [C,H,W])
        levels: Number of pyramid levels
        low_boost: Factor to boost low frequency (overall brightness)
        high_boost: Factor to boost high frequencies (details)

    Returns:
        Enhanced image of the same type as input
    """
    # Create pyramid decomposer
    pyramid = LaplacianPyramid(num_levels=levels)

    # Decompose image
    low_freq, high_freqs = pyramid.decompose(image)

    # Boost low frequency (brightness)
    if isinstance(low_freq, torch.Tensor):
        boosted_low = low_freq * low_boost
        # Boost high frequencies (details)
        boosted_high = [hf * high_boost for hf in high_freqs]
    else:
        boosted_low = low_freq * low_boost
        # Boost high frequencies (details)
        boosted_high = [hf * high_boost for hf in high_freqs]

    # Reconstruct enhanced image
    enhanced = pyramid.reconstruct(boosted_low, boosted_high)

    return enhanced, pyramid, low_freq, high_freqs, boosted_low, boosted_high


def visualize_enhancement(original, enhanced, title="Dark Image Enhancement"):
    """Visualize original and enhanced images side by side"""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Ensure images are in correct format for display
    if isinstance(original, torch.Tensor):
        if original.dim() == 3:  # [C,H,W]
            original_vis = original.permute(1, 2, 0).cpu().numpy()
        else:  # [B,C,H,W]
            original_vis = original[0].permute(1, 2, 0).cpu().numpy()
    else:
        if original.ndim == 3:  # [H,W,C]
            original_vis = original
        else:  # [B,H,W,C]
            original_vis = original[0]

    if isinstance(enhanced, torch.Tensor):
        if enhanced.dim() == 3:  # [C,H,W]
            enhanced_vis = enhanced.permute(1, 2, 0).cpu().numpy()
        else:  # [B,C,H,W]
            enhanced_vis = enhanced[0].permute(1, 2, 0).cpu().numpy()
    else:
        if enhanced.ndim == 3:  # [H,W,C]
            enhanced_vis = enhanced
        else:  # [B,H,W,C]
            enhanced_vis = enhanced[0]

    # Display original image
    axes[0].imshow(cv2.cvtColor(np.clip(original_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Dark Image')
    axes[0].axis('off')

    # Display enhanced image
    axes[1].imshow(cv2.cvtColor(np.clip(enhanced_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def visualize_pyramid(image, low_freq, high_freqs, title="Laplacian Pyramid", pyramid=None):
    """Visualize the pyramid decomposition"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, len(high_freqs) + 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)

    # Ensure image is in correct format for display
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # [C,H,W]
            image_vis = image.permute(1, 2, 0).cpu().numpy()
        else:  # [B,C,H,W]
            image_vis = image[0].permute(1, 2, 0).cpu().numpy()
    else:
        if image.ndim == 3:  # [H,W,C]
            image_vis = image
        else:  # [B,H,W,C]
            image_vis = image[0]

    # First row: Original image and low frequency component
    axes[0, 0].imshow(cv2.cvtColor(np.clip(image_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Handle either numpy or tensor low_freq
    if isinstance(low_freq, torch.Tensor):
        if low_freq.dim() > 3:  # [B,C,H,W]
            low_freq_vis = low_freq[0].permute(1, 2, 0).cpu().numpy()
        else:  # [C,H,W]
            low_freq_vis = low_freq.permute(1, 2, 0).cpu().numpy()
    else:
        if low_freq.ndim > 3:  # [B,H,W,C]
            low_freq_vis = low_freq[0]
        else:  # [H,W,C]
            low_freq_vis = low_freq

    # Show low frequency component
    axes[0, 1].imshow(cv2.cvtColor(np.clip(low_freq_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Low Frequency')
    axes[0, 1].axis('off')

    # Show high frequency components
    for i, hf in enumerate(high_freqs):
        if isinstance(hf, torch.Tensor):
            if hf.dim() > 3:  # [B,C,H,W]
                hf_vis = hf[0].permute(1, 2, 0).cpu().numpy()
            else:  # [C,H,W]
                hf_vis = hf.permute(1, 2, 0).cpu().numpy()
        else:
            if hf.ndim > 3:  # [B,H,W,C]
                hf_vis = hf[0]
            else:  # [H,W,C]
                hf_vis = hf

        # Normalize for better visualization (high freq components often have small values)
        hf_abs = np.abs(hf_vis)
        hf_normalized = hf_abs / (np.max(hf_abs) + 1e-8)

        axes[0, i + 2].imshow(cv2.cvtColor(hf_normalized, cv2.COLOR_BGR2RGB))
        axes[0, i + 2].set_title(f'High Freq {i + 1}')
        axes[0, i + 2].axis('off')

    # Second row: Manipulated examples
    if pyramid is None:
        pyramid = LaplacianPyramid(num_levels=len(high_freqs))

    # Example 1: Enhanced details (boost high frequencies)
    enhanced_hfs = [hf * 2.0 for hf in high_freqs]

    if isinstance(low_freq, torch.Tensor):
        reconstructed_enhanced = pyramid.reconstruct(low_freq, enhanced_hfs)
        if reconstructed_enhanced.dim() > 3:  # [B,C,H,W]
            reconstructed_enhanced_vis = reconstructed_enhanced[0].permute(1, 2, 0).cpu().numpy()
        else:  # [C,H,W]
            reconstructed_enhanced_vis = reconstructed_enhanced.permute(1, 2, 0).cpu().numpy()
    else:
        reconstructed_enhanced = pyramid.reconstruct(low_freq, enhanced_hfs)
        if reconstructed_enhanced.ndim > 3:  # [B,H,W,C]
            reconstructed_enhanced_vis = reconstructed_enhanced[0]
        else:  # [H,W,C]
            reconstructed_enhanced_vis = reconstructed_enhanced

    axes[1, 0].imshow(cv2.cvtColor(np.clip(reconstructed_enhanced_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Enhanced Details (2x)')
    axes[1, 0].axis('off')

    # Example 2: Smoothed image (reduce high frequencies)
    smoothed_hfs = [hf * 0.5 for hf in high_freqs]

    if isinstance(low_freq, torch.Tensor):
        reconstructed_smoothed = pyramid.reconstruct(low_freq, smoothed_hfs)
        if reconstructed_smoothed.dim() > 3:  # [B,C,H,W]
            reconstructed_smoothed_vis = reconstructed_smoothed[0].permute(1, 2, 0).cpu().numpy()
        else:  # [C,H,W]
            reconstructed_smoothed_vis = reconstructed_smoothed.permute(1, 2, 0).cpu().numpy()
    else:
        reconstructed_smoothed = pyramid.reconstruct(low_freq, smoothed_hfs)
        if reconstructed_smoothed.ndim > 3:  # [B,H,W,C]
            reconstructed_smoothed_vis = reconstructed_smoothed[0]
        else:  # [H,W,C]
            reconstructed_smoothed_vis = reconstructed_smoothed

    axes[1, 1].imshow(cv2.cvtColor(np.clip(reconstructed_smoothed_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Smoothed (0.5x)')
    axes[1, 1].axis('off')

    # Example 3: Tone mapping (boost low freq)
    if isinstance(low_freq, torch.Tensor):
        boosted_low = low_freq * 1.5
        reconstructed_tone = pyramid.reconstruct(boosted_low, high_freqs)
        if reconstructed_tone.dim() > 3:  # [B,C,H,W]
            reconstructed_tone_vis = reconstructed_tone[0].permute(1, 2, 0).cpu().numpy()
        else:  # [C,H,W]
            reconstructed_tone_vis = reconstructed_tone.permute(1, 2, 0).cpu().numpy()
    else:
        boosted_low = low_freq * 1.5
        reconstructed_tone = pyramid.reconstruct(boosted_low, high_freqs)
        if reconstructed_tone.ndim > 3:  # [B,H,W,C]
            reconstructed_tone_vis = reconstructed_tone[0]
        else:  # [H,W,C]
            reconstructed_tone_vis = reconstructed_tone

    axes[1, 2].imshow(cv2.cvtColor(np.clip(reconstructed_tone_vis, 0, 1), cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Boosted Brightness (1.5x)')
    axes[1, 2].axis('off')

    # Fill remaining plots with band-specific manipulations
    for i in range(min(3, len(high_freqs))):
        modified_hfs = [hf.clone() if isinstance(hf, torch.Tensor) else hf.copy() for hf in high_freqs]
        # Boost only one frequency band
        modified_hfs[i] = modified_hfs[i] * 3.0

        if isinstance(low_freq, torch.Tensor):
            band_enhanced = pyramid.reconstruct(low_freq, modified_hfs)
            if band_enhanced.dim() > 3:  # [B,C,H,W]
                band_enhanced_vis = band_enhanced[0].permute(1, 2, 0).cpu().numpy()
            else:  # [C,H,W]
                band_enhanced_vis = band_enhanced.permute(1, 2, 0).cpu().numpy()
        else:
            band_enhanced = pyramid.reconstruct(low_freq, modified_hfs)
            if band_enhanced.ndim > 3:  # [B,H,W,C]
                band_enhanced_vis = band_enhanced[0]
            else:  # [H,W,C]
                band_enhanced_vis = band_enhanced

        axes[1, i + 3].imshow(cv2.cvtColor(np.clip(band_enhanced_vis, 0, 1), cv2.COLOR_BGR2RGB))
        axes[1, i + 3].set_title(f'Enhanced Band {i + 1} (3x)')
        axes[1, i + 3].axis('off')

    # Hide any unused subplots
    for i in range(len(high_freqs) + 3, axes.shape[1]):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def create_synthetic_dark_image(h=512, w=512):
    """Create a synthetic dark tests image"""
    # Create a base image with some structure
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # Create circular gradient
    cx, cy = 0.5, 0.5  # center
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Create a synthetic image with different elements
    image = np.zeros((h, w, 3), dtype=np.float32)

    # Add some structure (circles, gradients)
    image[:, :, 0] = np.clip(0.15 * (1 - radius), 0, 0.3)  # dark red channel
    image[:, :, 1] = np.clip(0.1 * (1 - radius) + 0.05 * np.sin(x * 10 * np.pi) * np.sin(y * 10 * np.pi), 0,
                             0.25)  # dark green
    image[:, :, 2] = np.clip(0.2 * (1 - radius ** 2), 0, 0.35)  # dark blue channel

    # Add some noise to simulate camera noise in dark images
    noise = np.random.normal(0, 0.01, image.shape)
    image = np.clip(image + noise, 0, 1)

    return image


def list_available_images(config):
    """List all available tests images from config"""
    if not config["dark_images"]:
        print("No images defined in config file.")
        return

    print("\nAvailable tests images:")
    for idx, (name, path) in enumerate(config["dark_images"].items()):
        print(f"{idx + 1}. {name}: {path}")


# Test with both NumPy and PyTorch inputs
if __name__ == "__main__":
    config_path = "image_paths.json"
    config = load_config(config_path)

    # Initialize config if it doesn't exist
    if "dark_images" not in config:
        config["dark_images"] = {}

    # Command line argument handling
    if len(sys.argv) > 1:
        # Check if user wants to add a new image to config
        if sys.argv[1] == "--add" and len(sys.argv) >= 4:
            image_name = sys.argv[2]
            image_path = sys.argv[3]
            if os.path.isfile(image_path):
                config["dark_images"][image_name] = image_path
                save_config(config, config_path)
                print(f"Added image '{image_name}' with path '{image_path}' to config")
            else:
                print(f"Error: File '{image_path}' does not exist")
            sys.exit(0)

        # Check if user wants to list available images
        elif sys.argv[1] == "--list":
            list_available_images(config)
            sys.exit(0)

        # Check if user wants to visualize the pyramid decomposition
        elif sys.argv[1] == "--pyramid" and len(sys.argv) > 2:
            pyramid_mode = True
            image_arg = sys.argv[2]
        else:
            pyramid_mode = False
            image_arg = sys.argv[1]

        # Check if user provided an image name from config
        if image_arg in config["dark_images"]:
            image_path = config["dark_images"][image_arg]
            print(f"Using image '{image_arg}' from config: {image_path}")

        # Check if user provided a direct path
        elif os.path.isfile(image_arg):
            image_path = image_arg
            print(f"Using provided image path: {image_path}")

        else:
            print(f"Error: Image '{image_arg}' not found in config and not a valid file path")
            list_available_images(config)
            print("\nUsing synthetic dark image instead")
            image = create_synthetic_dark_image()
    else:
        # If no arguments provided, let user choose from config
        list_available_images(config)
        if config["dark_images"]:
            choice = input("\nEnter the number of the image to use (or press Enter for synthetic): ")
            if choice.isdigit() and 1 <= int(choice) <= len(config["dark_images"]):
                image_name = list(config["dark_images"].keys())[int(choice) - 1]
                image_path = config["dark_images"][image_name]
                print(f"Using image '{image_name}': {image_path}")

                # Ask if user wants to see pyramid decomposition
                pyramid_mode = input(
                    "Do you want to visualize the Laplacian pyramid decomposition? (y/n): ").lower() == 'y'
            else:
                print("Using synthetic dark image")
                image = create_synthetic_dark_image()
                pyramid_mode = input(
                    "Do you want to visualize the Laplacian pyramid decomposition? (y/n): ").lower() == 'y'
        else:
            print("No images in config. Using synthetic dark image")
            image = create_synthetic_dark_image()
            pyramid_mode = input("Do you want to visualize the Laplacian pyramid decomposition? (y/n): ").lower() == 'y'

    # Load image if path is defined
    if 'image_path' in locals():
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            image = create_synthetic_dark_image()
            print("Using synthetic dark image instead")
        else:
            # Normalize image to [0, 1] range
            image = image.astype(np.float32) / 255.0

    # Process with NumPy input
    print("Processing with NumPy input...")
    enhanced_np, pyramid_np, low_freq_np, high_freqs_np, boosted_low_np, boosted_high_np = enhance_dark_image(
        image, levels=4, low_boost=1.8, high_boost=2.5
    )

    if pyramid_mode:
        # Visualize the pyramid decomposition
        visualize_pyramid(
            image, low_freq_np, high_freqs_np,
            title="Laplacian Pyramid Decomposition (NumPy)",
            pyramid=pyramid_np
        )

        # Also visualize the boosted pyramid components
        visualize_pyramid(
            image, boosted_low_np, boosted_high_np,
            title="Boosted Laplacian Pyramid (NumPy)",
            pyramid=pyramid_np
        )

    # Visualize the enhancement result
    visualize_enhancement(image, enhanced_np, "Dark Image Enhancement (NumPy)")

    # Process with PyTorch input
    print("Processing with PyTorch input...")
    # Convert image to PyTorch tensor [C, H, W]
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
    enhanced_pt, pyramid_pt, low_freq_pt, high_freqs_pt, boosted_low_pt, boosted_high_pt = enhance_dark_image(
        image_tensor, levels=4, low_boost=1.8, high_boost=2.5
    )

    if pyramid_mode:
        # Visualize the pyramid decomposition
        visualize_pyramid(
            image_tensor, low_freq_pt, high_freqs_pt,
            title="Laplacian Pyramid Decomposition (PyTorch)",
            pyramid=pyramid_pt
        )

        # Also visualize the boosted pyramid components
        visualize_pyramid(
            image_tensor, boosted_low_pt, boosted_high_pt,
            title="Boosted Laplacian Pyramid (PyTorch)",
            pyramid=pyramid_pt
        )

    # Visualize the enhancement result
    visualize_enhancement(image_tensor, enhanced_pt, "Dark Image Enhancement (PyTorch)")

    print("Processing completed!")
