"""
Program Name: Medial Axis Transform Algorithm
Author: Justin McGarr
Description: This module implements a distance-guided medial-axis
skeletonisation algorithm for binary fingerprint images.

The algorithm uses Euclidean distances and maximal-disk centres
to guide topology-preserving thinning.

Date: 10/10/2026
"""

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


# Eight neighbouring pixel positions
NEIGHBOURS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# Four neighbouring pixel positions
FOUR_NEIGHBOURS = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1)
]


def maximal_disk_centres(binary_img, distances):
    """
    Identifies the centres of discrete maximal inscribed disks.

    Each foreground pixel is considered a potential disk centre.
    Its radius is the Euclidean distance to the nearest background pixel.

    Disks contained within larger disks are discarded.
    """

    # Find coordinates of all foreground pixels
    points = np.argwhere(binary_img > 0)

    # Create an empty image for storing maximal-disk centres
    centres = np.zeros(binary_img.shape, dtype=bool)

    if len(points) == 0:
        return centres

    # Obtain the radius of each potential disk
    radii = distances[points[:, 0], points[:, 1]]

    # Sort disks from largest to smallest
    order = np.argsort(-radii, kind="stable")

    # Store accepted maximal disks
    accepted = []

    for index in order:

        y, x = map(int, points[index])
        radius = float(radii[index])

        contained = False

        # Compare current disk against previously accepted disks
        for ay, ax, other_radius in accepted:

            if other_radius < radius:
                continue

            # Calculate Euclidean distance between disk centres
            centre_distance = math.hypot(ay - y, ax - x)

            # Check whether the current disk is contained
            if centre_distance + radius <= other_radius + 1e-10:
                contained = True
                break

        # Retain disks that are not contained
        if not contained:
            centres[y, x] = True
            accepted.append((y, x, radius))

    return centres


def _components(patch, directions):
    """
    Counts connected groups of foreground pixels in a
    3x3 neighbourhood.

    Used to determine whether removing a pixel would
    affect connectivity.
    """

    visited = np.zeros((3, 3), dtype=bool)
    count = 0

    for y in range(3):
        for x in range(3):

            # Skip background or already visited pixels
            if not patch[y, x] or visited[y, x]:
                continue

            # A new connected component has been found
            count += 1

            stack = [(y, x)]
            visited[y, x] = True

            # Explore all connected pixels
            while stack:

                cy, cx = stack.pop()

                for dy, dx in directions:

                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < 3
                        and 0 <= nx < 3
                        and patch[ny, nx]
                        and not visited[ny, nx]
                    ):

                        visited[ny, nx] = True
                        stack.append((ny, nx))

    return count


def _simple_point(image, y, x):
    """
    Checks whether a pixel can be removed without
    changing the local topology of the image.

    Returns True if the pixel can safely be removed.
    """

    # Extract the 3x3 neighbourhood
    patch = image[y - 1:y + 2, x - 1:x + 2].copy()

    # Preserve background pixels, isolated pixels and endpoints
    if not patch[1, 1] or np.count_nonzero(patch) <= 2:
        return False

    # Simulate removing the centre pixel
    patch[1, 1] = False

    # Check foreground and background connectivity
    return (
        _components(patch, NEIGHBOURS) == 1
        and _components(~patch, FOUR_NEIGHBOURS) == 1
    )


def reconstruct_skeleton(binary_img, distances, centres):
    """
    Generates a digital skeleton using distance-guided thinning.

    Boundary pixels are removed in order of increasing
    Euclidean distance.

    Maximal-disk centres are given lower removal priority,
    but can still be removed when topologically safe.
    """

    # Convert image to Boolean and add a background border
    image = np.pad(
        binary_img > 0,
        1,
        constant_values=False
    )

    # Pad the distance map and maximal-disk centres
    distance = np.pad(
        distances,
        1,
        constant_values=0
    )

    centres = np.pad(
        centres,
        1,
        constant_values=False
    )

    height, width = image.shape

    # Priority queue for boundary pixels
    queue = []

    # Track pixels already in the queue
    queued = np.zeros_like(image)

    def enqueue(y, x):
        """
        Adds an eligible boundary pixel to the priority queue.
        """

        # Skip background pixels or pixels already queued
        if not image[y, x] or queued[y, x]:
            return

        # Check whether the pixel is on the boundary
        if any(
            not image[y + dy, x + dx]
            for dy, dx in NEIGHBOURS
        ):

            # Process smaller distances first
            # For equal distances, process non-centre pixels first
            heapq.heappush(
                queue,
                (
                    float(distance[y, x]),
                    int(centres[y, x]),
                    y,
                    x
                )
            )

            queued[y, x] = True

    # Add initial boundary pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            enqueue(y, x)

    # Continue until no more pixels are queued
    while queue:

        # Retrieve the next boundary pixel
        _, _, y, x = heapq.heappop(queue)

        queued[y, x] = False

        # Skip pixels already removed
        if not image[y, x]:
            continue

        # Remove the pixel if connectivity is preserved
        if _simple_point(image, y, x):

            image[y, x] = False

            # Check neighbouring pixels for removal
            for dy, dx in NEIGHBOURS:

                ny = y + dy
                nx = x + dx

                if (
                    1 <= ny < height - 1
                    and 1 <= nx < width - 1
                ):
                    enqueue(ny, nx)

    # Convert back to 0/255 binary format
    skeleton = image[1:-1, 1:-1].astype(np.uint8) * 255

    return skeleton


def medial_axis_thinning(binary_img: np.ndarray) -> np.ndarray:
    """
    Performs distance-guided medial-axis skeletonisation.

    Input:
        Binary image containing white foreground ridges (255)
        and black background pixels (0).

    Output:
        Skeletonised image with the same dimensions and format.
    """

    image = np.asarray(binary_img)

    # Check that the image is two-dimensional
    if image.ndim != 2:
        raise ValueError("Input must be a 2D image")

    # Check that the image contains only binary values
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Input must contain only 0 and 255")

    # Return an empty image if no foreground pixels exist
    if not np.any(image):
        return np.zeros(image.shape, dtype=np.uint8)

    # Step 1: Calculate the Euclidean Distance Transform
    foreground = np.pad(
        image > 0,
        1,
        constant_values=False
    )

    distances = distance_transform_edt(foreground)[1:-1, 1:-1]

    # Step 2: Identify maximal-disk centres
    centres = maximal_disk_centres(image, distances)

    # Step 3: Generate the digital skeleton
    skeleton = reconstruct_skeleton(
        image,
        distances,
        centres
    )

    return skeleton