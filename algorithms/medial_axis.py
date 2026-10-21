
"""
Medial Axis Transform (MAT) skeletonisation for binary fingerprint images.

Stages:
    1. Calculate Euclidean distances to the background.
    2. Identify discrete maximal-disk centres.
    3. Thin the image while protecting those centres and preserving local topology.

Input:  2D binary image, white ridges (255), black background (0).
Output: 2D uint8 skeleton image, white (255) on black (0).

Note: Protecting every maximal-disk centre does not guarantee a
one-pixel-wide skeleton. This is the same limitation as the previous version.
"""

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


# Neighbour offsets used for connectivity checks.
NEIGHBOURS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

FOUR_NEIGHBOURS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
]


def maximal_disk_centres(binary_img, distances):
    """
    Find foreground pixels whose Euclidean disks are not contained
    within a previously accepted disk of greater or equal radius.

    The radius of a disk is the distance to the nearest background pixel.
    This is a discrete pixel-centre approximation of maximal disks.
    """

    points = np.argwhere(np.asarray(binary_img) > 0)
    centres = np.zeros(np.asarray(binary_img).shape, dtype=bool)

    if len(points) == 0:
        return centres

    radii = distances[points[:, 0], points[:, 1]]

    # Check larger disks first so smaller, contained disks can be discarded.
    order = np.argsort(-radii, kind="stable")
    accepted = []

    for index in order:
        y, x = map(int, points[index])
        radius = float(radii[index])
        contained = False

        for ay, ax, other_radius in accepted:
            if other_radius - radius < 0:
                continue

            # A disk is contained when centre distance + its radius
            # does not exceed the radius of the larger disk.
            if math.hypot(ay - y, ax - x) <= other_radius - radius + 1e-10:
                contained = True
                break

        if not contained:
            centres[y, x] = True
            accepted.append((y, x, radius))

    return centres


def _components(patch, directions):
    """
    Count connected groups of True pixels in a 3x3 neighbourhood.
    """

    visited = np.zeros((3, 3), dtype=bool)
    count = 0

    for y in range(3):
        for x in range(3):
            if not patch[y, x] or visited[y, x]:
                continue

            count += 1
            stack = [(y, x)]
            visited[y, x] = True

            # Flood-fill one connected group.
            while stack:
                cy, cx = stack.pop()

                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx

                    if (0 <= ny < 3 and 0 <= nx < 3
                            and patch[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

    return count


def _simple_point(image, y, x):
    """
    Check whether a pixel can be removed without changing local topology.
    """

    patch = image[y - 1:y + 2, x - 1:x + 2].copy()

    # Preserve background pixels, isolated pixels and endpoints.
    if not patch[1, 1] or np.count_nonzero(patch) <= 2:
        return False

    # Simulate removal, then test foreground and background connectivity.
    patch[1, 1] = False

    return (
        _components(patch, NEIGHBOURS) == 1
        and _components(~patch, FOUR_NEIGHBOURS) == 1
    )


def reconstruct_skeleton(binary_img, distances, centres):
    """
    Thin boundary pixels in distance order, retaining maximal-disk centres.

    Uses a priority queue so pixels nearer the background are tried first.
    Only locally topology-preserving deletions are performed.
    """

    image = np.pad(np.asarray(binary_img) > 0, 1, constant_values=False)
    protected = np.pad(centres, 1, constant_values=False)
    distance = np.pad(distances, 1, constant_values=0)

    height, width = image.shape
    queue = []
    queued = np.zeros_like(image)

    def enqueue(y, x):
        """
        Queue an unprotected boundary pixel if it is not queued already.
        """

        if not image[y, x] or protected[y, x] or queued[y, x]:
            return

        # Boundary pixels have at least one background neighbour.
        if any(not image[y + dy, x + dx] for dy, dx in NEIGHBOURS):
            heapq.heappush(queue, (float(distance[y, x]), y, x))
            queued[y, x] = True

    # Start with the boundary pixels of the original binary image.
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            enqueue(y, x)

    while queue:
        _, y, x = heapq.heappop(queue)
        queued[y, x] = False

        if not image[y, x] or protected[y, x]:
            continue

        if _simple_point(image, y, x):
            image[y, x] = False

            # Removing a pixel can expose new boundary pixels.
            for dy, dx in NEIGHBOURS:
                ny, nx = y + dy, x + dx

                if 1 <= ny < height - 1 and 1 <= nx < width - 1:
                    enqueue(ny, nx)

    return image[1:-1, 1:-1].astype(np.uint8) * 255


def medial_axis_thinning(binary_img: np.ndarray) -> np.ndarray:
    """
    Return a distance-guided MAT skeleton of a 0/255 binary image.
    """

    image = np.asarray(binary_img)

    if image.ndim != 2:
        raise ValueError("Input must be a 2D image")

    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Input must contain only 0 and 255")

    if not np.any(image):
        return np.zeros(image.shape, dtype=np.uint8)

    # Padding makes the outside of the image count as background.
    foreground = np.pad(image > 0, 1, constant_values=False)
    distances = distance_transform_edt(foreground)[1:-1, 1:-1]

    # Identify the centres of maximal inscribed disks.
    centres = maximal_disk_centres(image, distances)

    # Generate the digital skeleton.
    return reconstruct_skeleton(image, distances, centres)
