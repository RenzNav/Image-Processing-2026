"""
Program Name: Pipeline
Author: Damisola Talabi
Description: Generic pipeline runner. Takes an algorithm name, processes all
             sample fingerprints with that algorithm, saves skeletons to disk,
             extracts minutiae, and registers everything in the JSON database.
Usage:
    python pipeline.py morphological_thinning
    python pipeline.py zhang_suen
    python pipeline.py medial_axis
    python pipeline.py all
"""
import sys
import os
import time
import math
import importlib
import cv2

import database as db
from preprocessing import load_grayscale, preprocess
from matching import build_template

# Map algorithm name -> (module path, function name)
ALGORITHMS = {
    "morphological_thinning": ("algorithms.morphological", "morphological_thinning"),
    "zhang_suen":             ("algorithms.zhang_suen",     "zhang_suen_thinning"),
    "medial_axis":            ("algorithms.medial_axis",    "medial_axis_thinning"),
}

MINUTIA_TYPES = {1: "ridge_ending", 3: "bifurcation"}

SAMPLES = [
    ("fp_001", "subject_001", "fingerprint_samples/fingerprint_01.jpg"),
    ("fp_002", "subject_002", "fingerprint_samples/fingerprint_02.jpg"),
]


def run_algorithm(algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(ALGORITHMS)}")

    module_name, func_name = ALGORITHMS[algorithm]
    module = importlib.import_module(module_name)
    algo_fn = getattr(module, func_name)

    out_dir = f"imgDATA/processed/{algorithm}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {algorithm} ===")

    for fid, sid, path in SAMPLES:
        gray = load_grayscale(path)
        binary = preprocess(gray)

        start = time.perf_counter()
        skeleton = algo_fn(binary)
        elapsed_ms = (time.perf_counter() - start) * 1000

        skel_path = f"{out_dir}/{fid}_skeleton.png"
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

        db.register_output(fid, sid, path, algorithm, skel_path, features)
        db.save_metric(fid, algorithm, elapsed_ms, minutiae_count=len(features))

        print(f"  {fid}: {elapsed_ms:>8.1f} ms | {len(features):>3} minutiae -> {skel_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <algorithm|all>")
        print("Algorithms:", list(ALGORITHMS))
        sys.exit(1)

    target = sys.argv[1]
    if target == "all":
        for alg in ALGORITHMS:
            run_algorithm(alg)
    else:
        run_algorithm(target)