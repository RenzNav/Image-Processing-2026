import cv2
import time
import os
import math
import database as db
from preprocessing import load_grayscale, preprocess
from algorithms.morphological import morphological_thinning
from matching import build_template

ALGORITHM = "morphological_thinning"
MINUTIA_TYPES = {1: "ridge_ending", 3: "bifurcation"}

os.makedirs(f"imgDATA/processed/{ALGORITHM}", exist_ok=True)

SAMPLES = [
    ("fp_001", "subject_001", "fingerprint_samples/fingerprint_01.jpg"),
    ("fp_002", "subject_002", "fingerprint_samples/fingerprint_02.jpg"),
]

for fid, sid, path in SAMPLES:
    gray = load_grayscale(path)
    binary = preprocess(gray)

    start = time.perf_counter()
    skeleton = morphological_thinning(binary)
    elapsed_ms = (time.perf_counter() - start) * 1000

    skel_path = f"imgDATA/processed/{ALGORITHM}/{fid}_skeleton.png"
    cv2.imwrite(skel_path, skeleton)

    template = build_template(skeleton)
    features = [
        {
            "type": MINUTIA_TYPES[int(m[3])],
            "x": int(m[0]),
            "y": int(m[1]),
            "orientation_deg": round(math.degrees(m[2]), 1),
        }
        for m in template["minutiae"]
    ]

    db.register_output(fid, sid, path, ALGORITHM, skel_path, features)
    db.save_metric(fid, ALGORITHM, elapsed_ms, minutiae_count=len(features))

    print(f"{fid}: {elapsed_ms:.1f} ms, {len(features)} minutiae -> {skel_path}")