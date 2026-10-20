"""
Program Name: Main
Author: Clarence Navarro
Description: This is the main program for the fingerprint skeletonisation project.
Date: 10/10/2026
"""

# Libraries needed to open a file dialogue and display results
import tkinter as tk
from tkinter import filedialog

# Library needed to display image results
import matplotlib.pyplot as plt 
import cv2

# Importing files we created for preprocessing, skeletonisation, and matching
from preprocessing import load_grayscale, preprocess
from algorithms.medial_axis import medial_axis_thinning
from algorithms.morphological import morphological_thinning
from algorithms.zhang_suen import zhang_suen_thinning
from matching import match_fingerprint, build_template, compare_templates, show_minutiae

# Directory where the fingerprint database is stored
# DATABASE_DIR = "database" 

# Algorithms available for skeletonisation
# Just comment out the ones you don't want to use
ALGORITHMS = {
    # "Medial Axis": medial_axis_thinning,
    # "Morphological Thinning": morphological_thinning,
    "Zhang-Suen": zhang_suen_thinning
}

# Function to open the file dialogue and select a fingerprint image
def select_image():
    root = tk.Tk()
    root.withdraw() # Hide the main window
    path = filedialog.askopenfilename(
        title="Select a fingerprint image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    root.destroy() # Close the file dialogue
    return path


def main():
    # Select the image and preprocess it
    path = select_image()
    if not path:
        print("No image selected. Exiting.")
        return
    gray = load_grayscale(path)
    binary = preprocess(gray) # Ridges = 255, background = 0


    # Run each thinning algorithm on the binary image 
    skeletons = {}
    for name, algorithm in ALGORITHMS.items():
        # Run the algorithm and store the result
        try:
            skeletons[name] = algorithm(binary.copy())
        except NotImplementedError:
            print(name, "algorithm not implemented yet. Skipping.")


    # Match the skeletons against the database (use first skeleton found)
    if skeletons:
        skeleton = list(skeletons.values())[0]
        print("Select the second fingerprint to compare against...")
        path_b = select_image()
        if path_b:
            algorithm = list(ALGORITHMS.values())[0]
            skeleton_b = algorithm(preprocess(load_grayscale(path_b)))

            tpl_a = build_template(skeleton)
            tpl_b = build_template(skeleton_b)
            score, pairs = compare_templates(tpl_a, tpl_b)
            show_minutiae(skeleton, tpl_a, "Fingerprint A")
            show_minutiae(skeleton_b, tpl_b, "Fingerprint B")
            print(f"Minutiae: A = {len(tpl_a['minutiae'])}, B = {len(tpl_b['minutiae'])}")
            print(f"Score = {score:.3f}, paired minutiae = {pairs}")
        # Call the match_fingerprint function and print the results
        # try:
            # print(match_fingerprint(skeleton, DATABASE_DIR))
        # except NotImplementedError:
           # print("Matching algorithm not implemented yet. Skipping.")


    # Display the original, binary, and skeleton images
    plt.figure(figsize=(4 * len(skeletons) + 2, 4))

    # Original grayscale image
    plt.subplot(1, len(skeletons) + 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')

    # Preprocessed binary image
    plt.subplot(1, len(skeletons) + 2, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Preprocessed Binary")
    plt.axis('off')

    # Results from each skeletonisation algorithm
    for i, (name, skeleton) in enumerate(skeletons.items(), start=3):
        plt.subplot(1, len(skeletons) + 2, i)
        plt.imshow(skeleton, cmap='gray')
        plt.title(name)
        plt.axis('off')

    # Adjust the spacing and display the images
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()