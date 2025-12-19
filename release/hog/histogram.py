"""
Part 1: HOG Feature Extraction - Cell Histogram Computation
Implement TODO 3.
"""

import numpy as np


def compute_cell_histogram(magnitude, angle, cell_size=8, nbins=9):
    """
    Compute gradient orientation histogram for each cell.
    
    Args:
        magnitude: (H, W) gradient magnitude
        angle: (H, W) gradient direction (0~180 degrees)
        cell_size: size of each cell (default: 8x8)
        nbins: number of histogram bins (default: 9)
    
    Returns:
        cell_histograms: (n_cells_y, n_cells_x, nbins) histogram for each cell
    
    Hint:
        - Divide the image into cell_size x cell_size cells
        - Create orientation histogram for pixels within each cell
        - Use bilinear interpolation to distribute magnitude to adjacent bins
    """
    
    height, width = magnitude.shape
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size
    
    # Array to store results
    cell_histograms = np.zeros((n_cells_y, n_cells_x, nbins))
    
    # ========================================
    # TODO 3: Implement cell histogram computation
    # ========================================
    
    # compute bin size
    bin_size = 180.0 / nbins

    # quantize pixels to lower/upper bin indices + weights (vectorized)
    bin_idx = angle / bin_size
    lower_bin = np.floor(bin_idx).astype(int)
    upper_bin = (lower_bin + 1) % nbins
    weight_upper = bin_idx - lower_bin
    weight_lower = 1.0 - weight_upper

    # iterate every cell
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start, y_end = cy * cell_size, (cy + 1) * cell_size
            x_start, x_end = cx * cell_size, (cx + 1) * cell_size

            # extract magnitude and bin info for the cell
            mag_cell = magnitude[y_start:y_end, x_start:x_end]
            lb_cell = lower_bin[y_start:y_end, x_start:x_end]
            ub_cell = upper_bin[y_start:y_end, x_start:x_end]
            wl_cell = weight_lower[y_start:y_end, x_start:x_end]
            wu_cell = weight_upper[y_start:y_end, x_start:x_end]

            # compute histogram (vectorized accumulation)
            for b in range(nbins):
                mask_l = lb_cell == b
                mask_u = ub_cell == b
                cell_histograms[cy, cx, b] = (
                    np.sum(mag_cell[mask_l] * wl_cell[mask_l]) +
                    np.sum(mag_cell[mask_u] * wu_cell[mask_u])
                )
    
    # ========================================
    
    return cell_histograms