"""
Part 3: Multi-Scale Sliding Window Detection - Image Pyramid Generation
Implement TODO 9.
"""

import numpy as np


def generate_pyramid(image, scale_factor=1.2, min_size=(64, 128), max_scale=1.5):
    """
    Generate an image pyramid for multi-scale detection.

    Args:
        image: (H, W, 3) input color image
        scale_factor: ratio between consecutive pyramid levels (default: 1.2)
        min_size: (min_width, min_height) minimum size to generate (default: 64x128)
        max_scale: maximum upscaling factor (default: 2.0)
                   Limits how much we upscale to avoid excessive computation
                   Common range: 1.5-2.5x for pedestrian detection

    Returns:
        pyramid: list of tuples (scaled_image, scale)
                 where scale is the scaling factor applied to the original image

    Hint:
        - For upscaling: stop when scale would exceed max_scale
        - For downscaling: stop when dimensions would be smaller than min_size
        - min_size is typically set as the window size
        - Use `resize_image` function below
    """

    # ========================================
    # TODO 9: Implement image pyramid generation
    # ========================================
    
    height, width = image.shape[:2]
    pyramid = []

    # start from original image
    scale = 1.0
    current_image = image.copy()
    pyramid.append((current_image, scale))

    # downscaling
    next_scale = scale / scale_factor
    next_height = int(height * next_scale)
    next_width = int(width * next_scale)

    # downscaling loop
    while next_width >= min_size[0] and next_height >= min_size[1]:
        scale = next_scale

        # resize
        current_image = resize_image(image, (next_height, next_width))
        pyramid.append((current_image, scale))

        # caclulate next iteration
        next_scale = scale / scale_factor
        next_height = int(height * next_scale)
        next_width = int(width * next_scale)

    # upscaling
    scale = 1.0
    next_scale = scale * scale_factor

    # upscaling loop
    while next_scale <= max_scale:
        scale = next_scale
        next_height = int(height * scale)
        next_width = int(width * scale)

        # resize
        current_image = resize_image(image, (next_height, next_width))
        pyramid.append((current_image, scale))

        # calculate next iteration
        next_scale = scale * scale_factor

    # sort pyramid from smallest to largest scale
    pyramid.sort(key=lambda x: x[1])
    
    # ========================================

    return pyramid



def resize_image(image, new_size):
    """
    Resizes an image using bilinear interpolation, using only NumPy.

    Args:
        image: (H, W, C) input NumPy array.
        new_size: (new_height, new_width) target size tuple.

    Returns:
        resized_image: (new_height, new_width, C) resized image.
    """
    
    old_height, old_width, n_channels = image.shape
    new_height, new_width = int(new_size[0]), int(new_size[1])
    
    resized = np.zeros((new_height, new_width, n_channels), dtype=image.dtype)
    
    # Ensure float division for ratios
    row_ratio = float(old_height) / float(new_height)
    col_ratio = float(old_width) / float(new_width)

    # Use float64 for precise calculations
    image_float = image.astype(np.float64)

    for i in range(new_height):
        for j in range(new_width):
            
            # Map coordinates from new image to original
            src_row = i * row_ratio
            src_col = j * col_ratio
            
            row_floor = int(np.floor(src_row))
            col_floor = int(np.floor(src_col))

            # Find the 4 neighboring pixels (P11, P12, P21, P22)
            # and clip indices to stay within image boundaries
            r1 = min(row_floor,     old_height - 1)
            c1 = min(col_floor,     old_width - 1)
            r2 = min(row_floor + 1, old_height - 1)
            c2 = min(col_floor + 1, old_width - 1)

            p11 = image_float[r1, c1] # Top-Left
            p12 = image_float[r1, c2] # Top-Right
            p21 = image_float[r2, c1] # Bottom-Left
            p22 = image_float[r2, c2] # Bottom-Right

            # Calculate fractional parts for weights
            row_frac = src_row - row_floor
            col_frac = src_col - col_floor

            # Calculate bilinear interpolation weights
            w1 = (1 - row_frac) * (1 - col_frac)
            w2 = (1 - row_frac) * col_frac
            w3 = row_frac * (1 - col_frac)
            w4 = row_frac * col_frac
            
            # Compute weighted sum for all channels
            interpolated_pixel = w1 * p11 + w2 * p12 + w3 * p21 + w4 * p22
            
            resized[i, j] = interpolated_pixel.astype(image.dtype)
            
    return resized