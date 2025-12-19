"""
Part 3: Sliding Window Detection - Object Detector
Integrate image pyramid, sliding window, and NMS for multi-scale object detection.
Implement TODO 12.
"""

from .pyramid import generate_pyramid
from .sliding_window import sliding_window
from .nms import non_maximum_suppression

import numpy as np
from tqdm import tqdm


class ObjectDetector:
    """
    Multi-scale object detector using HOG features and SVM classifier.

    This detector integrates:
    1. Image pyramid for multi-scale detection
    2. Sliding window for exhaustive scanning
    3. HOG feature extraction for each window
    4. SVM classification for detection
    5. Non-maximum suppression for final detections

    Parameters:
        descriptor: HOG feature extractor
        classifier: trained SVM classifier (LinearSVM instance)
        window_size: (width, height) detection window size (default: 64x128)
        scale_factor: pyramid scale factor (default: 1.2)
        step_size: sliding window step size (default: 8)
        score_threshold: minimum detection score (default: 0.0)
        iou_threshold: NMS IoU threshold (default: 0.3)
    """

    def __init__(
        self,
        descriptor,
        classifier,
        window_size=(64, 128),
        scale_factor=1.2,
        step_size=8,
        score_threshold=0.1,
        iou_threshold=0.3
    ):
        self.descriptor = descriptor
        self.classifier = classifier
        
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.step_size = step_size
        
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold


    def detect(self, image, verbose=False):
        """
        Detect objects in one or multiple images using multi-scale sliding window approach.

        Args:
            images: a single (H, W, 3) image
            verbose: whether to print detection progress (default: False)

        Returns:
            final_detections: list of detections (bboxes)
            each detection is a tuple (x, y, width, height, score)
            where (x, y) is top-left corner in original image coordinates

        Hint:
            - Generate image pyramid for each image
            - Apply sliding window detection at each pyramid scale
            - Apply non-maximum suppression
            - Return final detections
        """

        # ========================================
        # TODO 12: Implement Pedestrian Detection pipeline
        # ========================================
        
        all_detections = []

        # generate image pyramid
        pyramid = generate_pyramid(
            image,
            scale_factor=self.scale_factor,
            min_size=self.window_size
        )

        # process each scale in the pyramid
        for scaled_image, scale in (tqdm(pyramid) if verbose else pyramid):
            
            # slide window across scaled image
            for (x, y, window) in sliding_window(
                scaled_image,
                window_size=self.window_size,
                step_size=self.step_size
            ):
                # extract HOG feature
                features = self.descriptor.extract(window)

                # get SVM decision score
                score = self.classifier.predict([features])[0]

                # filter by score threshold
                if score > self.score_threshold:
                    # rescale using scale factor to original image coordinates
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(self.window_size[0] / scale)
                    orig_h = int(self.window_size[1] / scale)

                    all_detections.append((orig_x, orig_y, orig_w, orig_h, score))

        # apply Non Maximum Suppression
        final_detections = non_maximum_suppression(
            all_detections,
            iou_threshold=self.iou_threshold
        )
        
        # ========================================

        return final_detections


    def set_threshold(self, score_threshold=None, iou_threshold=None):
        """
        Update detection thresholds.

        Args:
            score_threshold: minimum detection score (optional)
            iou_threshold: NMS IoU threshold (optional)
        """
        if score_threshold is not None:
            self.score_threshold = score_threshold

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
