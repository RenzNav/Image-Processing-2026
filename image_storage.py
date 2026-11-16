"""
Program Name: Image Storage
Author: Clarence Navarro
Description: This module is called within the thinning algorithms to save
processed images to their respective processed directories. It ensures that 
the output images are saved in the correct format and location. 
The process includes:
    - Receive processed image
    - Identify algorithm that produced the image
    - Locate the corresponding processed directory
    - Check if the image already exists
    - If exists, skip saving
    - If it doesn't exist, save image
Date: 12/11/2026
"""

import cv2
import numpy as np
from pathlib import Path

def save_processed_image(image, fingerprint_id, algorithm):
    """
    Saves a processed fingerprint image in the corresponding
    algorithm's processed directory.

    Checks whether the image already exists before saving.
    """

    # Directory where processed fingerprint images are stored
    processed_dir = Path("imgDATA/processed") / algorithm

    # Create the directory if it does not exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create the filename for the processed fingerprint
    filename = f"{fingerprint_id}_skeleton.png"

    # Get the full path where the image will be saved
    image_path = processed_dir / filename

    # Check if the processed image already exists
    if image_path.exists():
        print(f"{filename} already exists. Skipping.")
        return image_path

    # Save the processed image
    if cv2.imwrite(str(image_path), image):
        print(f"Saved {filename} to {processed_dir}")
        return image_path

    print(f"Failed to save {filename}")
    return None

if __name__ == "__main__":

    # Create a small test image
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[30:70, 30:70] = 255

    # Save the test image
    save_processed_image(test_image, "test_fingerprint", "zhang_suen")