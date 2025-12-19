"""
Part 1: HOG Feature Extraction - HOG Descriptor
Implement TODO 4 and TODO 5.
"""

import numpy as np
from .gradient import compute_gradient, compute_magnitude_angle
from .histogram import compute_cell_histogram


class HOGDescriptor:
    """
    HOG (Histogram of Oriented Gradients) feature descriptor.
    
    Parameters:
        cell_size: size of each cell (default: 8x8 pixels)
        block_size: size of each block (default: 2x2 cells)
        nbins: number of bins in gradient orientation histogram (default: 9)
    """
    
    def __init__(self, cell_size=8, block_size=2, nbins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        
    def normalize_block(self, cell_histograms):
        """
        Perform block normalization.
        
        Args:
            cell_histograms: (n_cells_y, n_cells_x, nbins) histogram for each cell
        
        Returns:
            features: (n_features,) normalized HOG feature vector
        
        Hint:
            - Group cells into block_size x block_size blocks
            - Concatenate histograms within each block and apply L2-normalization
            - Flatten all normalized block features into a 1D vector
        """
        
        n_cells_y, n_cells_x, nbins = cell_histograms.shape
        
        # Calculate number of blocks
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        
        # Total feature dimension
        # Each block: block_size^2 * nbins dimensions
        feature_dim = n_blocks_y * n_blocks_x * (self.block_size ** 2) * nbins
        features = np.zeros(feature_dim)
        
        epsilon = 1e-5  # Small value for numerical stability

        # ========================================
        # TODO 4: Implement block normalization
        # ========================================

        feature_idx = 0

        # iterate each block
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                # extract histogram
                block_hist = cell_histograms[
                    by:by + self.block_size,
                    bx:bx + self.block_size
                ].flatten()
                
                # L2-Hys normalization
                block_norm = np.sqrt(np.sum(block_hist ** 2) + epsilon ** 2)
                block_normalized = block_hist / block_norm

                # clip by 0.2
                block_normalized = np.clip(block_normalized, 0, 0.2)

                # normalize again
                block_norm = np.sqrt(np.sum(block_normalized ** 2) + epsilon ** 2)
                block_normalized = block_normalized / block_norm

                # store the normalized block features
                features[feature_idx:feature_idx + len(block_hist)] = block_normalized
                feature_idx += len(block_hist)
        
        # ========================================
        
        return features
    
    def extract(self, image):
        """
        Extract HOG features from an image.
        
        Args:
            image: (H, W, 3) color image
        
        Returns:
            features: (n_features,) HOG feature vector
        """
        
        # ========================================
        # TODO 5: Implement HOG feature extraction
        # ========================================
        
        # compute gradients
        gx, gy = compute_gradient(image)
        
        # compute magnitude and angle
        magnitude, angle = compute_magnitude_angle(gx, gy)
        
        # compute cell histograms
        cell_histograms = compute_cell_histogram(
            magnitude, angle,
            cell_size=self.cell_size,
            nbins=self.nbins
        )
        
        # apply block normalization to get final features
        features = self.normalize_block(cell_histograms)
        features = features.astype(np.float32)
        
        # ========================================
        
        return features
    


    def get_cell_hists(self, image):
        """
        Extract HOG features from an image.
        
        Args:
            image: (H, W, 3) color image
        
        Returns:
            features: (n_features,) HOG feature vector
        """
        # 1. Compute gradients
        # Gradients are computed per channel and max magnitude is selected
        gx, gy = compute_gradient(image)
        
        # 2. Compute magnitude and angle
        magnitude, angle = compute_magnitude_angle(gx, gy)
        
        # 3. Compute cell histograms
        cell_histograms = compute_cell_histogram(
            magnitude, angle, 
            cell_size=self.cell_size, 
            nbins=self.nbins
        )
        
        return cell_histograms
    