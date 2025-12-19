"""
Part 2: SVM Training - Gradient Computation
Implement TODO 7.
"""

import numpy as np


def compute_gradient(X, y, w, b, C=1.0):
    """
    Compute the gradient of hinge loss with respect to w and b.
    
    Args:
        X: (N, D) feature matrix
        y: (N,) labels (+1 or -1)
        w: (D,) weight vector
        b: scalar bias term
        C: regularization parameter (default: 1.0)
    
    Returns:
        grad_w: (D,) gradient with respect to w
        grad_b: scalar gradient with respect to b
    
    Hint:
        - For samples with margin >= 1: no gradient contribution
        - Don't forget to add regularization gradient: w (for grad_w)
    """
    
    N, D = X.shape

    # ========================================
    # TODO 7: Implement gradient computation
    # ========================================
    
    # calculate margins for all samples
    margins = y * (np.dot(X, w) + b)

    # identify samples that violate the margin (margin < 1)
    mask = margins < 1

    # data term gradients (only when it violates margin)
    # ∂L/∂w = w - C * (1/N) * Σ(y_i * x_i)
    grad_w = w - C * np.mean(X[mask] * y[mask, np.newaxis], axis=0)

    # ∂L/∂b = -C * (1/N) * Σ(y_i) for margin<1
    grad_b = -C * np.mean(y[mask])

    # ========================================
    
    return grad_w, grad_b