import cv2
import time
import os
import database as db
from preprocessing import load_grayscale, preprocess
from algorithms.zhang_suen import zhang_suen_thinning
from DBmatching import store_minutiae, match_against_database

ALGORITHM = "zhang_suen"
os.makedirs("imgDATA/processed/zhang_suen", exist_ok=True)

SAMPLES = [
    ("fp_001", "fingerprint_samples/fingerprint_01.jpg"),
    ("fp_002", "fingerprint_samples/fingerprint_02.jpg"),
]

for fid, path in SAMPLES:
    gray = load_grayscale(path)
    binary = preprocess(gray)

    start = time.perf_counter()
    skeleton = zhang_suen_thinning(binary)
    elapsed_ms = (time.perf_counter() - start) * 1000

    skel_path = f"imgDATA/processed/zhang_suen/{fid}_skeleton.png"
    cv2.imwrite(skel_path, skeleton)
    db.update_processed_images(fid, ALGORITHM, skeletonised=skel_path)
    db.save_metric(fid, ALGORITHM, elapsed_ms, minutiae_count=0)

    print(f"{fid}: {elapsed_ms:.1f} ms -> {skel_path}")

for fid, _ in SAMPLES:
    n = store_minutiae(fid, ALGORITHM)
    print(f"{fid} minutiae: {n}")

print("Match result:")
print(match_against_database("fp_001", ALGORITHM))