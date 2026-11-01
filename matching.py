"""
Program Name: Matching Algorithm
Author: Justin McGarr
Description:  
Date: 
"""
import numpy as np
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------------
# Tunable parameters (pixel values assume the size preprocessing resizes to)
# ----------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
 
BORDER_MARGIN = 10        # px: ignore minutiae this close to the print edge
MIN_MINUTIA_DIST = 8      # px: minutiae closer than this are spurs/breaks -> removed
TRACE_STEPS = 10          # px: how far to follow a ridge to get its orientation
 
HOUGH_ANGLE_BIN = 10      # degrees
HOUGH_POS_BIN = 8         # px
HOUGH_TOP_K = 20           # number of candidate alignments to verify
 
PAIR_DIST_TOL = 12        # px: max distance for two minutiae to be a pair
PAIR_ANGLE_TOL = 20       # degrees: max orientation difference for a pair
 
MIN_DENOMINATOR = 12      # stops tiny overlaps from scoring ~100%
 
_template_cache = {}
 
 
# ----------------------------------------------------------------------------
# 1. Minutiae extraction (crossing number method)
# ----------------------------------------------------------------------------
_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
 
 
def _crossing_number(skel):
    h, w = skel.shape
    s = np.pad(skel.astype(np.int8), 1)
    ring = [s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _OFFSETS]
    cn = np.zeros((h, w), dtype=np.int16)
    for i in range(8):
        cn += np.abs(ring[i] - ring[(i + 1) % 8])
    return cn // 2
 
 
def _region_masks(skel):
    """
    Returns (region, safe):
      region - area covered by the print (used for overlap checks)
      safe   - region shrunk by BORDER_MARGIN (minutiae outside are edge artefacts)
    """
    pad = 30
    padded = np.pad(skel.astype(np.uint8), pad)
    r = 10
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    region = cv2.dilate(padded, k_dil)
    k_ero = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * (r + BORDER_MARGIN) + 1,) * 2)
    safe = cv2.erode(region, k_ero)
    return region[pad:-pad, pad:-pad] > 0, safe[pad:-pad, pad:-pad] > 0
 
 
def _neighbours(skel, y, x):
    h, w = skel.shape
    out = []
    for dy, dx in _OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
            out.append((ny, nx))
    return out
 
 
def _trace_angle(skel, y, x, start, steps=TRACE_STEPS):
    """Follow the ridge from (y, x) via `start`; return angle to the end point."""
    visited = {(y, x)}
    visited.update(n for n in _neighbours(skel, y, x) if n != start)
    cur = start
    for _ in range(steps):
        visited.add(cur)
        nxt = [n for n in _neighbours(skel, *cur) if n not in visited]
        if not nxt:
            break
        cur = nxt[0]
    return math.atan2(cur[0] - y, cur[1] - x)
 
 
def _ang_diff(a, b):
    """Smallest absolute difference between angles (radians), works on arrays."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
 
 
def extract_minutiae(skel, safe):
    """
    Returns an (N, 4) array of [x, y, angle, type]; type 1 = ending, 3 = bifurcation.
    """
    cn = _crossing_number(skel)
    candidates = np.argwhere(skel & safe & ((cn == 1) | (cn == 3)))
 
    minutiae = []
    for y, x in candidates:
        branches = _neighbours(skel, y, x)
        if not branches:
            continue
        if cn[y, x] == 1:
            angle = _trace_angle(skel, y, x, branches[0])
            minutiae.append([x, y, angle, 1])
        else:
            if len(branches) < 3:
                continue
            angles = np.array([_trace_angle(skel, y, x, b) for b in branches])
            # The "stem" is the branch furthest from the other two
            spread = [sum(_ang_diff(a, o) for o in angles) for a in angles]
            minutiae.append([x, y, float(angles[int(np.argmax(spread))]), 3])
 
    if not minutiae:
        return np.zeros((0, 4))
    m = np.array(minutiae, dtype=np.float64)
 
    # Remove spurs / ridge breaks: any minutia with a close neighbour
    d = np.hypot(m[:, None, 0] - m[None, :, 0], m[:, None, 1] - m[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return m[d.min(axis=1) >= MIN_MINUTIA_DIST]
 
 
def build_template(skeleton):
    """Skeleton (uint8, ridges = 255) -> dict with minutiae + region mask."""
    skel = np.asarray(skeleton)
    if skel.ndim == 3:
        skel = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
    skel = skel > 127          # tolerant of JPEG artefacts in saved skeletons
    region, safe = _region_masks(skel)
    return {"minutiae": extract_minutiae(skel, safe), "region": region}
 
 
# ----------------------------------------------------------------------------
# 2. Alignment (Hough voting) and scoring
# ----------------------------------------------------------------------------
def _rotate(xy, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([c * xy[:, 0] - s * xy[:, 1],
                     s * xy[:, 0] + c * xy[:, 1]], axis=1)
 
 
def _hough_candidates(q, t):
    """Vote for (rotation, tx, ty); return the top-K transforms."""
    same_type = q[:, None, 3] == t[None, :, 3]
    qi, ti = np.nonzero(same_type)
    if len(qi) == 0:
        return []
 
    dth = (t[ti, 2] - q[qi, 2]) % (2 * np.pi)
    c, s = np.cos(dth), np.sin(dth)
    qx, qy = q[qi, 0], q[qi, 1]
    tx = t[ti, 0] - (c * qx - s * qy)
    ty = t[ti, 1] - (s * qx + c * qy)
 
    abin = (np.degrees(dth) // HOUGH_ANGLE_BIN).astype(int)
    keys = np.stack([abin, (tx // HOUGH_POS_BIN).astype(int),
                     (ty // HOUGH_POS_BIN).astype(int)], axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
 
    n = len(counts)
    sin_sum = np.bincount(inv, np.sin(dth), n)
    cos_sum = np.bincount(inv, np.cos(dth), n)
    tx_mean = np.bincount(inv, tx, n) / counts
    ty_mean = np.bincount(inv, ty, n) / counts
    theta_mean = np.arctan2(sin_sum, cos_sum)
 
    top = np.argsort(-counts)[:HOUGH_TOP_K]
    return [(theta_mean[i], tx_mean[i], ty_mean[i]) for i in top]
 
 
def _pair_up(q, t, theta, tx, ty):
    """Apply the transform to q and greedily pair minutiae."""
    q_xy = _rotate(q[:, :2], theta) + np.array([tx, ty])
    q_ang = q[:, 2] + theta
 
    d = np.hypot(q_xy[:, None, 0] - t[None, :, 0], q_xy[:, None, 1] - t[None, :, 1])
    ad = _ang_diff(q_ang[:, None], t[None, :, 2])
    ok = (d < PAIR_DIST_TOL) & (ad < math.radians(PAIR_ANGLE_TOL)) \
        & (q[:, None, 3] == t[None, :, 3])
 
    cand = np.argwhere(ok)
    cand = cand[np.argsort(d[cand[:, 0], cand[:, 1]])]
    used_q, used_t, pairs = set(), set(), []
    for i, j in cand:
        if i not in used_q and j not in used_t:
            used_q.add(i)
            used_t.add(j)
            pairs.append((i, j))
    return pairs
 
 
def _count_inside(xy, region):
    h, w = region.shape
    x = np.round(xy[:, 0]).astype(int)
    y = np.round(xy[:, 1]).astype(int)
    inb = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    inside = np.zeros(len(xy), dtype=bool)
    inside[inb] = region[y[inb], x[inb]]
    return int(inside.sum())
 
 
def _evaluate(q, t, q_region, t_region, theta, tx, ty):
    pairs = _pair_up(q, t, theta, tx, ty)
 
    # One refinement step: re-estimate the transform from the matched pairs
    if len(pairs) >= 3:
        qi = np.array([p[0] for p in pairs])
        ti = np.array([p[1] for p in pairs])
        dth = t[ti, 2] - q[qi, 2]
        theta2 = math.atan2(np.sin(dth).mean(), np.cos(dth).mean())
        shifted = t[ti, :2] - _rotate(q[qi, :2], theta2)
        tx2, ty2 = shifted.mean(axis=0)
        pairs2 = _pair_up(q, t, theta2, tx2, ty2)
        if len(pairs2) >= len(pairs):
            pairs, theta, tx, ty = pairs2, theta2, tx2, ty2
            
 
    matched = len(pairs)
 
    # Only count minutiae that fall inside the other print's area, so a small
    # partial print can still score highly against a full one.
    q_xy = _rotate(q[:, :2], theta) + np.array([tx, ty])
    q_overlap = _count_inside(q_xy, t_region)
    t_in_q = _rotate(t[:, :2] - np.array([tx, ty]), -theta)
    t_overlap = _count_inside(t_in_q, q_region)
 
    # Overlap-based Dice similarity
    overlap_total = q_overlap + t_overlap

    if overlap_total == 0:
        score = 0.0
    else:
        overlap_score = min(1.0, (2 * matched) / overlap_total)

    # Penalise matches supported by very few minutiae
    MIN_EVIDENCE = 12
    evidence_factor = min(1.0, matched / MIN_EVIDENCE)

    # Final similarity score
    score = overlap_score * evidence_factor

    return score, matched, pairs, (theta, tx, ty)
 
 

def compare_templates(a, b, return_details=False):
    """Compare two fingerprint templates."""

    q, t = a["minutiae"], b["minutiae"]

    if len(q) < 3 or len(t) < 3:
        if return_details:
            return 0.0, 0, [], None
        return 0.0, 0

    best = (0.0, 0, [], None)

    for theta, tx, ty in _hough_candidates(q, t):
        res = _evaluate(
            q, t,
            a["region"], b["region"],
            theta, tx, ty
        )

        if res[0] > best[0] or (
            res[0] == best[0] and res[1] > best[1]
        ):
            best = res

    if return_details:
        return best

    return best[0], best[1]

def show_minutiae(skeleton, template, title):
    minutiae = template["minutiae"]

    plt.figure(figsize=(7, 7))
    plt.imshow(skeleton, cmap="gray")

    endings = minutiae[minutiae[:, 3] == 1]
    bifurcations = minutiae[minutiae[:, 3] == 3]

    plt.scatter(
        endings[:, 0],
        endings[:, 1],
        color="red",
        s=12,
        label="Ridge endings"
    )

    plt.scatter(
        bifurcations[:, 0],
        bifurcations[:, 1],
        color="blue",
        s=12,
        label="Bifurcations"
    )

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()
 
 
# ----------------------------------------------------------------------------
# 3. Database handling + public entry point
# ----------------------------------------------------------------------------
def _load_database(database_dir):
    """Load every skeleton image in the folder and extract its minutiae (cached)."""
    files = sorted(f for f in os.listdir(database_dir)
                   if f.lower().endswith(IMAGE_EXTS))
    stamp = tuple((f, os.path.getmtime(os.path.join(database_dir, f))) for f in files)
    cached = _template_cache.get(database_dir)
    if cached and cached[0] == stamp:
        return cached[1]
 
    templates = {}
    for f in files:
        img = cv2.imread(os.path.join(database_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {f}: could not read image")
            continue
        templates[f] = build_template(img)
    _template_cache[database_dir] = (stamp, templates)
    return templates

# Input: 2D uint8array, foreground (ridges) = 255, background = 0
# Need to mention database directory after Dami creates it
# Expected Output: List of tuples showing the best match from the database?

def match_fingerprint(skeleton: np.ndarray, database_dir: str, top_n: int = 5) -> list:
    query = build_template(skeleton)
    database = _load_database(database_dir)
 
    results = []
    for name, tpl in database.items():
        score, n = compare_templates(query, tpl)
        results.append((name, round(float(score), 3), n))
    results.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return results[:top_n]
