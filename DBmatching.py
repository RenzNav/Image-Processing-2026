"""
Program Name: Database Matching
Author: Damisola Talabi
Description: Connects the matching code (matching.py) to the JSON database (database.py).
             It loads the skeleton of each enrolled fingerprint using the paths stored in
             the database, compares them with a query fingerprint, and saves the minutiae
             and the match results back into the database.
Date: 
"""
import math
import cv2
import database as db
from matching import build_template, compare_templates



# matching.py stores the minutia type as a number (1 or 3); the database stores a word.
MINUTIA_TYPES = {1: "ridge_ending", 3: "bifurcation"}


def _load_template(fingerprint_id, algorithm):
    record = db.get_fingerprint(fingerprint_id)
    if record is None:
        raise ValueError(f"{fingerprint_id} is not in the database")
    path = record["processed_images"].get("skeletonised", {}).get(algorithm)
    if not path:
        raise ValueError(f"{fingerprint_id} has no {algorithm} skeleton")
    skeleton = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        raise FileNotFoundError(f"Could not read skeleton: {path}")
    return build_template(skeleton)


def store_minutiae(fingerprint_id, algorithm):
    template = _load_template(fingerprint_id, algorithm)
    features = [
        {
            "type": MINUTIA_TYPES[int(m[3])],
            "x": int(m[0]),
            "y": int(m[1]),
            "orientation_deg": round(math.degrees(m[2]), 1),
        }
        for m in template["minutiae"]
    ]
    db.save_minutiae(fingerprint_id, algorithm, features)
    return len(features)


def match_against_database(query_id, algorithm, top_n=5):
    query = _load_template(query_id, algorithm)
    scored = []
    for record in db.get_all_fingerprints():
        other_id = record["fingerprint_id"]
        if other_id == query_id:
            continue
        try:
            other = _load_template(other_id, algorithm)
        except (ValueError, FileNotFoundError):
            continue
        score, matched = compare_templates(query, other)
        scored.append((other_id, float(score), matched))
    scored.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return [db.save_match_result(query_id, oid, round(s, 3), algorithm)
            for oid, s, _ in scored[:top_n]]