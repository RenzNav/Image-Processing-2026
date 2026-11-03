"""Bridge between the database and matching.py."""
import cv2
import database as db
from matching import match_fingerprint


def run_match(query_fid, algorithm, skeleton_dir, top_n=5):
    fp = db.get_fingerprint(query_fid)
    if fp is None:
        raise ValueError(f"Unknown fingerprint: {query_fid}")

    skel_path = fp["processed_images"]["skeletonised"]
    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise ValueError(f"Could not read skeleton: {skel_path}")

    results = match_fingerprint(skel, skeleton_dir)

    for filename, score, _n in results:
        matched_fid = db.find_by_skeleton_filename(filename)
        if matched_fid is None or matched_fid == query_fid:
            continue
        db.save_match_result(query_fid, matched_fid, score, algorithm)

    return results


if __name__ == "__main__":
    out = run_match("fp_001", "morphological_thinning", "imgDATA/processed")
    for r in out:
        print(r)