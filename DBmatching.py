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

# Algorithm name recorded next to the minutiae we save.
ALGORITHM = "morphological_thinning"

# matching.py stores the minutia type as a number (1 or 3); the database stores a word.
MINUTIA_TYPES = {1: "ridge_ending", 3: "bifurcation"}


def _load_template(fingerprint_id):
    """Look up a fingerprint in the database, read its skeleton image and build its template."""
    record = db.get_fingerprint(fingerprint_id)
    if record is None:
        raise ValueError(f"{fingerprint_id} is not in the database")

    # The database only stores the PATH of the skeleton, so the image is read from disk here.
    path = record["processed_images"].get("skeletonised")
    if not path:
        raise ValueError(f"{fingerprint_id} has no skeletonised image saved in the database")

    skeleton = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        raise FileNotFoundError(f"Could not read skeleton image: {path}")

    # build_template extracts the minutiae and the print region from the skeleton.
    return build_template(skeleton)


def store_minutiae(fingerprint_id, algorithm=ALGORITHM):
    """Extract the minutiae of a fingerprint's skeleton and save them in minutiae.json.
    Returns how many minutiae were saved."""
    template = _load_template(fingerprint_id)

    # Convert each row [x, y, angle in radians, type] into the dictionary format
    # that the database expects.
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


def match_against_database(query_id, top_n=5):
    """Compare one enrolled fingerprint against every other enrolled fingerprint.
    The top_n best results are saved in results.json and returned."""
    query = _load_template(query_id)

    scored = []
    for record in db.get_all_fingerprints():
        other_id = record["fingerprint_id"]

        # Skip the query itself (it would always be a perfect match) and any
        # fingerprint that has no skeleton yet.
        if other_id == query_id or not record["processed_images"].get("skeletonised"):
            continue

        score, matched = compare_templates(query, _load_template(other_id))
        scored.append((other_id, float(score), matched))

    # Best score first; the number of matched minutiae breaks ties.
    scored.sort(key=lambda r: (r[1], r[2]), reverse=True)

    # Save the best results; save_match_result labels each one match / no_match.
    return [db.save_match_result(query_id, other_id, round(score, 3))
            for other_id, score, _ in scored[:top_n]]