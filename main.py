"""
Program Name: Main
Author: Clarence Navarro
Description: This is the main program for the fingerprint skeletonisation project.
Date: 10/10/2026
"""

# Libraries needed to open a file dialogue and display results
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importing files we created for preprocessing, skeletonisation, and matching
from preprocessing import load_grayscale, preprocess
# from algorithms.medial_axis import medial_axis_thinning
# from algorithms.morphological import morphological_thinning
from algorithms.zhang_suen import zhang_suen_thinning
from matching import match_fingerprint
from postprocessing import prune_spurs


DATABASE_DIR = "database" 

# Algorithms available for skeletonisation
# Just comment out the ones you don't want to use
ALGORITHMS = {
    # "Medial Axis": medial_axis_thinning,
    # "Morphological Thinning": morphological_thinning,
    "Zhang-Suen": zhang_suen_thinning
}


def select_image():
    """
    Opens a file dialogue for the user to select a fingerprint image. 
    """
    root = tk.Tk()
    # Hide the main window
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a fingerprint image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    # Close the file dialogue
    root.destroy() 
    return path


def main():
    """
    Main function to run the fingerprint skeletonisation project.
    It allows the user to select a fingerprint image, preprocesses it, applies multiple skeletonisation algorithms, and matches the results against a database.
    """
    # Select the image and preprocess it
    path = select_image()
    if not path:
        print("No image selected. Exiting.")
        return
    gray = load_grayscale(path)
    binary = preprocess(gray)


    # Run each thinning algorithm on the binary image 
    skeletons = {}
    processed_skeletons = {}
    for name, algorithm in ALGORITHMS.items():
        try:
            # Run the thinning algorithm
            skeletons[name] = algorithm(binary.copy())

            # Apply spur pruning to the skeleton
            processed_skeletons[name] = prune_spurs(
                skeletons[name], threshold=2
            )
        except NotImplementedError:
            print(name, "algorithm not implemented yet. Skipping.")


    # Match the skeletons against the database (use first skeleton found)
    if skeletons:
        skeleton = list(skeletons.values())[0]
        try:
            print(match_fingerprint(skeleton, DATABASE_DIR))
        except NotImplementedError:
            print("Matching algorithm not implemented yet. Skipping.")


    # Display the original, binary, skeleton, and post-processed images
    for name, skeleton in skeletons.items():

        # Get the post-processed skeleton
        processed = processed_skeletons[name]

        # Identify pixels removed during post-processing
        removed = (skeleton > 0) & (processed == 0)

        # Convert grayscale skeleton to RGB
        overlay = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        
        # Highlight removed pixels in red
        overlay[removed] = [255, 0, 0]

        plt.figure(figsize=(20, 5))

        # Display original grayscale image
        plt.subplot(1, 5, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Original Grayscale")
        plt.axis('off')

        # Display preprocessed binary image
        plt.subplot(1, 5, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Preprocessed Binary")
        plt.axis('off')

        # Display original results from each skeletonisation algorithm
        plt.subplot(1, 5, 3)
        plt.imshow(skeleton, cmap='gray', vmin=0, vmax=255)
        plt.title(name + " - Thinning")
        plt.axis('off')

        # Display postprocessed results from each skeletonisation algorithm
        plt.subplot(1, 5, 4)
        plt.imshow(processed, cmap='gray', vmin=0, vmax=255)
        plt.title(name + " - Post Processing")
        plt.axis('off')

        # Display removed pixels highlighted in red
        plt.subplot(1, 5, 5)
        plt.imshow(overlay)
        plt.title(name + " - Removed Pixels")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()