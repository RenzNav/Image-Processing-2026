"""
Program Name: Morphological Thinning Algorithm
Author: Damisola Talabi
Description: This module implements the morphological thinning algorithm for skeletonisation of binary images.
Date: 10/10/2026
"""
import numpy as np

# ---------------------------------------------------------------------------
# STRUCTURING ELEMENTS
# ---------------------------------------------------------------------------
# These are the 8 "templates" the algorithm slides over the image.
# Each template is a 3x3 grid with three possible values:
#    1  = this position must be foreground (255) for a match
#    0  = this position must be background (0) for a match
#   -1  = don't care (ignore this position)
#
# The centre pixel (position [1][1]) is the pixel under consideration, and it
# is always 1 because we only ever try to delete foreground pixels.
#
# Each template describes one kind of edge pixel: the zeros mark the side the
# background is on, and the ones mark the foreground that keeps the pixel
# connected to the rest of the shape. The first two templates (NORTH and
# NORTHEAST) are the base templates; the other six are the same two rotated
# in 90 degree steps. Together the 8 peel off a single layer of edge pixels
# from every direction, while preserving connectivity.
# ---------------------------------------------------------------------------

_TEMPLATES = [
    # 1: North — background above, foreground below
    np.array([[ 0,  0,  0],
              [-1,  1, -1],
              [ 1,  1,  1]], dtype=np.int8),
    # 2: Northeast
    np.array([[-1,  0,  0],
              [ 1,  1,  0],
              [-1,  1, -1]], dtype=np.int8),
    # 3: East
    np.array([[ 1, -1,  0],
              [ 1,  1,  0],
              [ 1, -1,  0]], dtype=np.int8),
    # 4: Southeast
    np.array([[-1,  1, -1],
              [ 1,  1,  0],
              [-1,  0,  0]], dtype=np.int8),
    # 5: South
    np.array([[ 1,  1,  1],
              [-1,  1, -1],
              [ 0,  0,  0]], dtype=np.int8),
    # 6: Southwest
    np.array([[-1,  1, -1],
              [ 0,  1,  1],
              [ 0,  0, -1]], dtype=np.int8),
    # 7: West
    np.array([[ 0, -1,  1],
              [ 0,  1,  1],
              [ 0, -1,  1]], dtype=np.int8),
    # 8: Northwest
    np.array([[ 0,  0, -1],
              [ 0,  1,  1],
              [-1,  1, -1]], dtype=np.int8),
]

# Split each template into the two sets the hit-or-miss transform uses:
#   B1 = positions that must be foreground (the 1s in the template)
#   B2 = positions that must be background (the 0s in the template)
# The -1 "don't care" positions end up in neither set, so they are ignored.
_STRUCTURING_ELEMENTS = [(T == 1, T == 0) for T in _TEMPLATES]


def _hit_or_miss(img: np.ndarray, B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """
    Hit-or-miss transform.
    Formula:  A ⊗ B = (A ⊖ B1) ∩ (A^c ⊖ B2)
    In words: "the pixel matches B1 on the foreground AND matches B2 on the
    background at the same time."
    """
    h, w = img.shape

    # Every pixel starts as "no match". We fill this in as we scan.
    result = np.ones((h - 2, w - 2), dtype=bool)

    # Loop over every INTERIOR pixel. We skip the outer border because a 3x3
    # template centred on a border pixel would run off the edge of the image.
    for dr in range(3):
        for dc in range(3):
            b1_val = B1[dr, dc]
            b2_val = B2[dr, dc]

            # The image region aligned with this template cell
            region = img[dr:dr + h - 2, dc:dc + w - 2]

            if b1_val == 1:
                result &= (region == 255)
            if b2_val == 1:
                result &= (region == 0)

    full = np.zeros((h, w), dtype=bool)
    full[1:-1, 1:-1] = result
    return full

def morphological_thinning(binary_img: np.ndarray) -> np.ndarray:
    """
    Skeletonise a binary image by repeated morphological thinning.

    Input:  2D uint8 array, ridges (foreground) = 255, background = 0.
    Output: 2D uint8 array, same shape, 1-pixel-wide skeleton = 255, background = 0.

    Algorithm: apply the 8 structuring elements in sequence. Each one removes
    a specific class of edge pixel. A full pass over all 8 strips one layer off
    the shape. Repeat until a full pass removes nothing — at that point the
    foreground is a skeleton.
    """
    # Normalise the input to a clean 0 / 255 uint8 array. Anything nonzero
    # becomes foreground. This makes the rest of the code's comparisons safe.
    img = (binary_img > 0).astype(np.uint8) * 255

    # Flag that tells us whether the last pass removed any pixels.
    # We keep looping as long as it stays True.
    changed = True

    
    while changed:
        

        # Assume this pass removes nothing. Any removal flips this to True.
        changed = False

        # Apply each of the 8 structuring elements in order. Crucially, each
        # element operates on the image as modified by the previous one, so
        # the shape shrinks progressively rather than all at once.
        for B1, B2 in _STRUCTURING_ELEMENTS:

            # Find every pixel currently matching this template.
            hits = _hit_or_miss(img, B1, B2)

            # If anything matched, delete those pixels (set them to background)
            # and record that the image changed this pass.
            if hits.any():
                img[hits] = 0
                changed = True

    # img now holds the skeleton. Return it in the same format as the input.
    return img