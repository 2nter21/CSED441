"""
Part 1: HOG Feature Extraction - Gradient Computation
Implement TODO 1 and TODO 2.
"""

import numpy as np


def compute_gradient(image):
    """
    Compute the x and y gradients of the image.
    
    Args:
        image: (H, W, 3) color image
    
    Returns:
        gx: (H, W) gradient in x direction
        gy: (H, W) gradient in y direction
    
    Hint:
        - For color images, compute gradients for each channel (R, G, B) and choose the maximum gradient
        - Use central difference method: [-1, 0, 1] kernel
        - You can use `np.pad()` with mode='edge' for boundary handling
    """
    
    image = image.astype(float)
    height, width, n_channels = image.shape
    
    # Initialize arrays to store the final gradients
    gx = np.zeros((height, width))
    gy = np.zeros((height, width))

    # ========================================
    # TODO 1: Implement gradient computation
    # ========================================
    
    # padding for central difference
    padded = np.pad(image, ((1,1), (1,1), (0,0)), mode='edge')

    # compute per-channel central differences (i.e. kernel [-0.5, 0, 0.5])
    gx_channels = 0.5 * (padded[1:-1, 2:, :] - padded[1:-1, :-2, :])
    gy_channels = 0.5 * (padded[2:, 1:-1, :] - padded[:-2, 1:-1, :])

    # magnitude per channel
    mag_channels = np.sqrt(gx_channels**2 + gy_channels**2)

    # pick channel with maximum magnitude at each pixel (vectorized, no python loops)
    idx = np.argmax(mag_channels, axis=2)                  # shape (H, W)
    rows, cols = np.indices((height, width))

    gx = gx_channels[rows, cols, idx]
    gy = gy_channels[rows, cols, idx]
    
    # ========================================
    
    return gx, gy


def compute_magnitude_angle(gx, gy):
    """
    Compute the magnitude and angle of gradients.
    
    Args:
        gx: (H, W) gradient in x direction
        gy: (H, W) gradient in y direction
    
    Returns:
        magnitude: (H, W) gradient magnitude
        angle: (H, W) gradient direction (0~180 degrees, unsigned)
    
    Hint:
        - You can use `np.degrees()` to convert angle from radians to degrees
        - For unsigned gradient, add 180 degree to negative angles
    """
    
    # ========================================
    # TODO 2: Implement magnitude and angle computation
    # ========================================
    
    # magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # calculate angle and convert between 0~180
    angle = np.degrees(np.arctan2(gy, gx))
    angle = np.mod(angle, 180.0)
    
    # ========================================
    
    return magnitude, angle