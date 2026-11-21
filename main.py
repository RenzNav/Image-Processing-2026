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

# Importing files we created for preprocessing, skeletonisation, postprocessing, processed images and matching
from preprocessing import load_grayscale, preprocess
from algorithms.medial_axis import medial_axis_thinning
from algorithms.morphological import morphological_thinning
from algorithms.zhang_suen import zhang_suen_thinning
from postprocessing import prune_spurs
from matching import match_fingerprint
from image_storage import save_processed_image

from pathlib import Path

DATABASE_DIR = Path(__file__).resolve().parent / "imgDATA"

# Algorithms available for skeletonisation
# Just comment out the ones you don't want to use
ALGORITHMS = {
    "Medial Axis Thinning": medial_axis_thinning,
    "Morphological Thinning": morphological_thinning,
    "Zhang-Suen Thinning": zhang_suen_thinning
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
    Allows the user to select fingerprint images, preprocess them,
    apply skeletonisation and spur pruning, compare fingerprint
    templates, and display the results.
    """
    # Select the image and preprocess it
    path = select_image()
    if not path:
        print("No image selected. Exiting.")
        return
    gray = load_grayscale(path)
    binary = preprocess(gray)

    # Use the selected image's filename as its fingerprint ID
    fingerprint_id = Path(path).stem

    # Choose whether to store or match the fingerprint
    mode = input("Choose mode (enroll/match): ").strip().lower()

    if mode not in ("enroll", "match"):
        print("Invalid mode. Choose 'enroll' or 'match'.")
        return

    # Run each thinning algorithm on the binary image 
    skeletons = {}
    processed_skeletons = {}

    MATCH_THRESHOLD = 0.5
    
    for name, algorithm in ALGORITHMS.items():
        try:
            # Run the thinning algorithm
            skeletons[name] = algorithm(binary.copy())

            # Apply spur pruning
            processed_skeletons[name] = prune_spurs(
                skeletons[name], threshold=2
            )

            # Save the post-processed image
            algorithm_key = name.lower().replace(" ", "_").replace("-", "_")

            # Only save fingerprints when enrolling
            if mode == "enroll":
                save_processed_image(
                    processed_skeletons[name],
                    fingerprint_id,
                    algorithm_key
                )

        except NotImplementedError:
            print(name, "algorithm not implemented yet. Skipping.")

    # Match each processed skeleton against the database
    if mode == "match":
        for name in skeletons:

            algorithm_key = name.lower().replace(" ", "_").replace("-", "_")

            print(f"\nMatching using {name}...")

            results = match_fingerprint(
                processed_skeletons[name],
                DATABASE_DIR,
                algorithm=algorithm_key
            )

            if not results:
                print(f"No database fingerprints found for {name}.")
                continue

            # Retrieve the best matching fingerprint
            matched_id, score, matched = results[0]

            print(f"Best match: {matched_id}")
            print(f"Similarity score: {score:.3f}")
            print(f"Matched minutiae: {matched}")

            if score >= MATCH_THRESHOLD:

                # Load the matched skeleton directly from the processed folder
                match_path = (
                    DATABASE_DIR / "processed" / algorithm_key
                    / f"{matched_id}_skeleton.png"
                )

                match_image = cv2.imread(
                    str(match_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if match_image is not None:
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(processed_skeletons[name], cmap="gray")
                    plt.title("Input Skeleton")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(match_image, cmap="gray")
                    plt.title(f"Best Match: {matched_id}\nScore: {score:.3f}")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()
            else:
                print("No sufficiently strong fingerprint match found.")


    # # Display the original, binary, skeleton, and post-processed images
    # for name, skeleton in skeletons.items():

    #     # Get the post-processed skeleton
    #     processed = processed_skeletons[name]

    #     # Identify pixels removed during post-processing
    #     removed = (skeleton > 0) & (processed == 0)

    #     # Convert grayscale skeleton to RGB
    #     overlay = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        
    #     # Highlight removed pixels in red
    #     overlay[removed] = [255, 0, 0]

    #     plt.figure(figsize=(20, 5))

    #     # Display original grayscale image
    #     plt.subplot(1, 5, 1)
    #     plt.imshow(gray, cmap='gray')
    #     plt.title("Original Grayscale")
    #     plt.axis('off')

    #     # Display preprocessed binary image
    #     plt.subplot(1, 5, 2)
    #     plt.imshow(binary, cmap='gray')
    #     plt.title("Preprocessed Binary")
    #     plt.axis('off')

    #     # Display original results from each skeletonisation algorithm
    #     plt.subplot(1, 5, 3)
    #     plt.imshow(skeleton, cmap='gray', vmin=0, vmax=255)
    #     plt.title(name + " - Thinning")
    #     plt.axis('off')

    #     # Display postprocessed results from each skeletonisation algorithm
    #     plt.subplot(1, 5, 4)
    #     plt.imshow(processed, cmap='gray', vmin=0, vmax=255)
    #     plt.title(name + " - Post Processing")
    #     plt.axis('off')

    #     # Display removed pixels highlighted in red
    #     plt.subplot(1, 5, 5)
    #     plt.imshow(overlay)
    #     plt.title(name + " - Removed Pixels")
    #     plt.axis('off')

    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    main()