"""
Program Name: Preprocessing
Author: Clarence Navarro
Description: This module is responsible for preprocessing fingerprint images to prepare them for skeletonisation.
It does the following:
    1. Converts the image to grayscale.
    2. Resizes the image to a standard size.
    3. Improves contrast between foreground (ridges) and background by applying histogram equalization. 
    4. Applies a small blur to reduce noise.
    5. Convert image to black and white.
    6. Removes small noise and fills minor gaps. DISABLED FOR NOW AS IT IS NOT WORKING WELL
    7. Removes small disconnected white regions not part of fingerprint ridges.
Output: Binary image with white fingerprint ridges and a black background 
Date: 10/10/2026
"""

import cv2
import numpy as np

# Change INVERT to true if the original image has light ridges on dark background.
INVERT = False


def load_grayscale(path):
    # 1. Convert the fingerprint image to grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read image: " + str(path))

    return image

# End load_grayscale


def preprocess(image):
    # 2. Resize the image to a standard size
    scale = 500 / max(image.shape) 

    image = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )


    # 3. Improve contrast between fingerprint ridges and background
    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )

    image = clahe.apply(image)


    # 4. Apply a small blur to reduce image noise
    image = cv2.GaussianBlur(image, (3, 3), 0)


    # 5. Convert grayscale image to black and white
    if INVERT:
        threshold_mode = cv2.THRESH_BINARY
    else:
        threshold_mode = cv2.THRESH_BINARY_INV

    binary = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_mode, 15, 2)


    # 6. Remove small noise and fill minor gaps
    # kernel = np.ones((3, 3), np.uint8)

    # binary = cv2.morphologyEx(
    #     # Remove small noise
    #     binary, cv2.MORPH_OPEN, kernel
    # )

    # binary = cv2.morphologyEx(
    #     # Fill minor gaps
    #     binary, cv2.MORPH_CLOSE, kernel
    # )


    # # 6. Remove small disconnected white regions
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    minimum_area = 10 # Minimum area threshold for connected components

    # Start from 1 to skip the background label (0)
    for component in range(1, count):
        area = stats[component, cv2.CC_STAT_AREA]

        # Set small components to background
        if area < minimum_area:
            binary[labels == component] = 0 


    return binary

# End preprocess