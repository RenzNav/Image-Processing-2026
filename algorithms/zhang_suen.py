"""
Program Name: Zhang-Suen Algorithm
Author: Clarence Navarro
Description: This module implements the Zhang-Suen thinning algorithm for skeletonisation of binary images. 
Date: 10/10/2026
"""
import numpy as np

def zhang_suen_thinning(binary_img: np.ndarray) -> np.ndarray:
    """
    Performs Zhang-Suen thinning on binary image.
    The input binary image has foreground pixel ridges represented by 255 and background pixels represented by 0. 
    Returns a skeletonised image with the same dimensions and format.
    """
    # Convert to binary for processing
    image = (binary_img > 0).astype(np.uint8)

    # Get height and width of image
    height, width = image.shape

    # Flag to track if any pixels are removed during an iteration
    changed = True

    # Continue iterating until no more pixels are removed
    while changed:

        # Reset flag at the start of each iteration
        changed = False

        # List for pixels marked for removal in each pass
        pixels_to_rem_p1 = []
        pixels_to_rem_p2 = []

        # Pass 1 of the algorithm to visit each pixel and check conditions for removal
        for row in range(1, height - 1): 
            for col in range(1, width - 1):   

                # Skip background pixels
                if  image[row, col] == 0:
                    continue

                # Store the 8 neighbors in clockwise order
                neighbors = [
                    image[row - 1, col],        # P2 - North
                    image[row - 1, col + 1],    # P3 - Northeast
                    image[row, col + 1],        # P4 - East
                    image[row + 1, col + 1],    # P5 - Southeast
                    image[row + 1, col],        # P6 - South
                    image[row + 1, col - 1],    # P7 - Southwest
                    image[row, col - 1],        # P8 - West
                    image[row - 1, col - 1]     # P9 - Northwest
                ]

                # Count the number of foreground (1) neighbors present
                fg_nbrs = sum(neighbors)

                # Condition 1: fg_nbrs must be between 2 and 6  
                if 2 <= fg_nbrs <= 6:

                    # Condition 2: Count the 0 to 1 transitions between neighbors
                    fg_trans_count = 0

                    for i in range(len(neighbors)):

                        # Check current neighbor, modulo used to go back to the first neighbor after the last
                        current = neighbors[i]
                        next_pixel = neighbors[(i + 1) % len(neighbors)]

                        # Increment counter for each 0 to 1 transition
                        if current == 0 and next_pixel == 1:
                            fg_trans_count += 1

                    # Conditions 3 and 4: Check the required neighbor combinations for Pass 1
                    if (fg_trans_count == 1
                            and neighbors[0] * neighbors[2] * neighbors[4] == 0
                            and neighbors[2] * neighbors[4] * neighbors[6] == 0):
                        
                        # Mark pixel for removal if all conditions met
                        pixels_to_rem_p1.append((row, col))
                
        # Remove marked pixels after scan is complete
        for row, col in pixels_to_rem_p1:
            image[row, col] = 0

        # Pass 2
        for row in range(1, height - 1): 
            for col in range(1, width - 1):   

                # Skip background pixels
                if  image[row, col] == 0:
                    continue

                # Store the 8 neighbors in clockwise order
                neighbors = [
                    image[row - 1, col],        # P2 - North
                    image[row - 1, col + 1],    # P3 - Northeast
                    image[row, col + 1],        # P4 - East
                    image[row + 1, col + 1],    # P5 - Southeast
                    image[row + 1, col],        # P6 - South
                    image[row + 1, col - 1],    # P7 - Southwest
                    image[row, col - 1],        # P8 - West
                    image[row - 1, col - 1]     # P9 - Northwest
                ]

                # Count the number of foreground (1) neighbors present
                fg_nbrs = sum(neighbors)

                # Condition 1: fg_nbrs must be between 2 and 6  
                if 2 <= fg_nbrs <= 6:

                    # Condition 2: Count the transitions from 0 to 1 between neighbors
                    fg_trans_count = 0

                    for i in range(len(neighbors)):
                        
                        # Check current neighbor, modulo used to go back to the first neighbor after the last
                        current = neighbors[i]
                        next_pixel = neighbors[(i + 1) % len(neighbors)]

                        # Increment counter for each 0 to 1 transition
                        if current == 0 and next_pixel == 1:
                            fg_trans_count += 1

                    # Conditions 3 and 4: Check the required neighbor combinations for Pass 2
                    if (fg_trans_count == 1
                        and neighbors[0] * neighbors[2] * neighbors[6] == 0
                        and neighbors[0] * neighbors[4] * neighbors[6] == 0):

                        # Mark pixel for removal if all conditions met
                        pixels_to_rem_p2.append((row, col))

        # Remove marked pixels after scan is complete
        for row, col in pixels_to_rem_p2:
            image[row, col] = 0

        # If any pixels were marked for removal in either pass, set changed to True
        if pixels_to_rem_p1 or pixels_to_rem_p2:
            changed = True

    # Convert back to original format: ridges = 255, background = 0 for return
    skeleton = (image * 255).astype(np.uint8)
    return skeleton

