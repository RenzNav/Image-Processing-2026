"""
Program Name: Post Processing
Author: Clarence Navarro
Description: This module implements spur pruning to remove short, unwanted branches
from skeletonised fingerprint images, while aiming to preserve the main ridge structures.
It includes:
    - Endpoint detection
    - Junction detection
    - Branch tracing
    - The main spur pruning function
Date: 22/10/2026
"""

import numpy as np

def prune_spurs(skeleton: np.ndarray, threshold: int = 2) -> np.ndarray:
    """
    Removes short branches connected to junctions.
    """

    # Convert to binary and create a separate copy
    image = (skeleton > 0).astype(np.uint8)

    # Find endpoints and junctions
    endpoints = find_endpoints(image)
    junctions = find_junctions(image)

    # Store pixels marked for removal
    pixels_to_remove = set()

    # Trace each branch
    for endpoint in endpoints:
        branch, reached_junction = trace_branch(image, endpoint, junctions)

        # Only remove confirmed short branches
        if reached_junction and len(branch) <= threshold:
            pixels_to_remove.update(branch)

    # Remove marked pixels
    for row, col in pixels_to_remove:
        image[row, col] = 0

    # Return skeleton in original 0/255 format
    return (image * 255).astype(np.uint8)


def find_endpoints(skeleton: np.ndarray):
    """
    Finds the endpoints of a skeletonised image.
    """
    endpoints = []

    # Convert to binary for processing
    image = (skeleton > 0).astype(np.uint8)
    height, width = image.shape

    # Visit every pixel, including border pixels
    for row in range(height):
        for col in range(width):

            # Skip background pixels
            if image[row, col] == 0:
                continue

            # Check the 8 neighbours of the pixel
            neighbors = check_neighbors(image, row, col)

            # Count the number of foreground (1) neighbours present
            fg_nbrs = sum(neighbors)

            # If the pixel has only one foreground neighbour, it is an endpoint
            if fg_nbrs == 1:
                endpoints.append((row, col))

    return endpoints


def find_junctions(skeleton: np.ndarray):
    """
    Finds the junctions of a skeletonised image.
    """
    junctions = []

    # Convert to binary for processing
    image = (skeleton > 0).astype(np.uint8)
    height, width = image.shape

    # Visit every pixel
    for row in range(height):
        for col in range(width):

            # Skip background pixels
            if image[row, col] == 0:
                continue

            # Check the 8 neighbours of the pixel
            neighbors = check_neighbors(image, row, col)

            # Count the number of 0-to-1 transitions
            transitions = 0

            for i in range(len(neighbors)):
                if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
                    transitions += 1

            # Three or more transitions indicate a possible junction
            if transitions >= 3:
                junctions.append((row, col))

    return junctions


def trace_branch(skeleton: np.ndarray, endpoint: tuple, junctions: list):
    """
    Traces a skeleton branch from an endpoint towards a junction.
    Returns the branch coordinates and whether a junction was reached.
    """

    # Convert to binary for processing
    image = (skeleton > 0).astype(np.uint8)

    # Store the coordinates of visited pixels
    branch = [endpoint]
    visited = {endpoint}

    # Start tracing from the endpoint
    current = endpoint
    junction_set = set(junctions)

    # Offsets for the 8 neighbouring pixels
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    height, width = image.shape

    while True:

        row, col = current
        next_pixels = []

        # Check each neighbouring pixel
        for dr, dc in directions:
            r = row + dr
            c = col + dc

            # Ensure the neighbour is within the image
            if 0 <= r < height and 0 <= c < width:
                # Check if the neighbour is foreground and has not been visited
                if image[r, c] == 1 and (r, c) not in visited:
                    next_pixels.append((r, c))

        # Check whether a junction has been reached
        if any(pixel in junction_set for pixel in next_pixels):
            return branch, True

        # Stop if the branch ends or becomes ambiguous
        if len(next_pixels) != 1:
            return branch, False

        # Move to the next pixel
        current = next_pixels[0]
        branch.append(current)
        visited.add(current)


def check_neighbors(image, row, col):
    """
    Checks the 8 neighbours of a pixel in a binary image.
    Returns a list of the values of the neighbours in clockwise order.
    """
    # Add padding to handle border pixels
    padded = np.pad(image, pad_width=1, mode="constant", constant_values=0)

    # Adjust coordinates for padded image
    r = row + 1
    c = col + 1

    # Store the 8 neighbours in clockwise order
    neighbors = [
        padded[r - 1, c],        # P2 - North
        padded[r - 1, c + 1],    # P3 - Northeast
        padded[r, c + 1],        # P4 - East
        padded[r + 1, c + 1],    # P5 - Southeast
        padded[r + 1, c],        # P6 - South
        padded[r + 1, c - 1],    # P7 - Southwest
        padded[r, c - 1],        # P8 - West
        padded[r - 1, c - 1]     # P9 - Northwest
    ]

    return neighbors