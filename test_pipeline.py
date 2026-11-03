import cv2
import time
import os
import database as db
from preprocessing import load_grayscale, preprocess
from algorithms.morphological import morphological_thinning

os.makedirs("imgDATA/processed", exist_ok=True)
os.makedirs("imgDATA/processed/binary", exist_ok=True)     # ← added

for i, path in enumerate(
    ["fingerprint_samples/fingerprint_01.jpg", "fingerprint_samples/fingerprint_02.jpg"],
    start=1,
):
    fid = f"fp_{i:03d}"

    db.add_fingerprint(fid, f"subject_{i:03d}", path, metadata={"format": "jpg"})

    gray = load_grayscale(path)
    binary = preprocess(gray)
    bin_path = f"imgDATA/processed/binary/{fid}_binary.png"    # ← changed
    cv2.imwrite(bin_path, binary)
    db.update_processed_images(fid, "morphological_thinning", binarised=bin_path)

    start = time.perf_counter()
    skeleton = morphological_thinning(binary)
    elapsed_ms = (time.perf_counter() - start) * 1000

    skel_path = f"imgDATA/processed/{fid}_skeleton.png"
    cv2.imwrite(skel_path, skeleton)
    db.update_processed_images(fid, "morphological_thinning", skeletonised=skel_path)

    db.save_metric(fid, "morphological_thinning", elapsed_ms, minutiae_count=0)

    print(f"{fid}: {elapsed_ms:.1f} ms → {skel_path}")