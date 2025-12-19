"""
Part 3: Multi-Scale Sliding Window Detection - Non-Maximum Suppression
Implement TODO 11.
"""

import numpy as np


def non_maximum_suppression(detections, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: list of tuples (x, y, width, height, score)
                    where (x, y) is top-left corner, score is confidence
        iou_threshold: IoU threshold for suppression (default: 0.3)
    
    Returns:
        kept_detections: list of tuples (x, y, width, height, score) after NMS
    
    Hint:
        - Sort detections by score in descending order
        - For each detection, remove all subsequent detections with IoU > threshold
        - Use the `compute_iou` function below
    """
    
    if len(detections) == 0:
        return []
    
    detections = np.array(detections)
    
    # ========================================
    # TODO 11: Implement Non-Maximum Suppression
    # ========================================
    
    detections = detections.astype(float)

    # sort detections by score in descending order
    indices = np.argsort(detections[:, 4])[::-1]
    detections = detections[indices]

    kept_detections = []

    # iterate until no boxes remain
    while len(detections) > 0:

        # pick the high-scored detection
        best_det = detections[0]
        kept_detections.append(tuple(best_det))

        # stop if no more detections remain
        if len(detections) == 1:
            break

        # compute IoU between the best detection and the others
        ious = np.array([compute_iou(best_det[:4], det[:4]) for det in detections[1:]])

        # keep only boxes with IoU <= threshold
        remaining = detections[1:][ious <= iou_threshold]

        # update the detections list
        detections = remaining
    
    # ========================================
    
    return kept_detections



def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: (x, y, width, height) first bounding box
        box2: (x, y, width, height) second bounding box
    
    Returns:
        iou: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Compute coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    if not union_area:
        return 0.0
    
    # Compute IoU
    iou = intersection_area / union_area
    
    return iou