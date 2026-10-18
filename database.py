"""
Program Name: Database (JSON storage layer)
Author: Dami Talabi
Description: JSON-backed storage for fingerprints, minutiae, matching results,
             and performance metrics. Replaces an earlier SQL design because
             JSON requires no server, is diff-friendly in git, and is
             appropriate for this project's small data volume.
Date: 18/10/2026
"""
import json
import os
from datetime import datetime

DATA_DIR = "imgDATA"  # Directory to store JSON files

# One file per category — keeps writes small and git diffs readable.
_FILES = {
    "fingerprints": os.path.join(DATA_DIR, "fingerprints.json"),
    "minutiae":     os.path.join(DATA_DIR, "minutiae.json"),
    "results":      os.path.join(DATA_DIR, "results.json"),
    "metrics":      os.path.join(DATA_DIR, "metrics.json"),
}


def _ensure_initialised():
    """Create the data directory and empty JSON files if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, path in _FILES.items():
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({key: []}, f, indent=2)


def _load(category):
    """Read a category's JSON file and return the list of records."""
    _ensure_initialised()
    with open(_FILES[category], "r") as f:
        return json.load(f)[category]


def _save(category, records):
    """Write the list of records back to the category's JSON file."""
    with open(_FILES[category], "w") as f:
        json.dump({category: records}, f, indent=2)



# Fingerprints
# -------------------
def add_fingerprint(fingerprint_id, subject_id, original_image_path, metadata=None):
    """Register a new fingerprint record."""
    records = _load("fingerprints")
    record = {
        "fingerprint_id": fingerprint_id,
        "subject_id": subject_id,
        "enrolled_date": datetime.now().isoformat(timespec="seconds"),
        "original_image_path": original_image_path,
        "metadata": metadata or {},
        "processed_images": {},
    }
    records.append(record)
    _save("fingerprints", records)
    return record


def get_fingerprint(fingerprint_id):
    """Return a fingerprint record by ID, or None."""
    for r in _load("fingerprints"):
        if r["fingerprint_id"] == fingerprint_id:
            return r
    return None


def update_processed_images(fingerprint_id, enhanced=None, binarised=None, skeletonised=None):
    """Attach paths to processed images for a fingerprint."""
    records = _load("fingerprints")
    for r in records:
        if r["fingerprint_id"] == fingerprint_id:
            if enhanced:
                r["processed_images"]["enhanced"] = enhanced
            if binarised:
                r["processed_images"]["binarised"] = binarised
            if skeletonised:
                r["processed_images"]["skeletonised"] = skeletonised
            break
    _save("fingerprints", records)



# Minutiae
# ---------------------
def save_minutiae(fingerprint_id, algorithm, features):
    """Store extracted minutiae for a fingerprint.
    features = list of dicts: {type, x, y, orientation_deg}
    """
    records = _load("minutiae")
    records.append({
        "fingerprint_id": fingerprint_id,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "features": features,
    })
    _save("minutiae", records)


def get_minutiae(fingerprint_id):
    """Return minutiae record for a fingerprint, or None."""
    for r in _load("minutiae"):
        if r["fingerprint_id"] == fingerprint_id:
            return r
    return None



# Matching results
# ----------------------
def save_match_result(query_id, matched_id, similarity_score, threshold=0.7):
    """Store a comparison result. Match status derived from threshold."""
    records = _load("results")
    status = "match" if similarity_score >= threshold else "no_match"
    record = {
        "comparison_id": f"cmp_{len(records) + 1:04d}",
        "query_fingerprint_id": query_id,
        "matched_fingerprint_id": matched_id,
        "similarity_score": similarity_score,
        "match_status": status,
        "compared_at": datetime.now().isoformat(timespec="seconds"),
    }
    records.append(record)
    _save("results", records)
    return record


# Performance metrics
# ---------------------------------
def save_metric(fingerprint_id, algorithm, execution_time_ms, minutiae_count, accuracy=None):
    """Record a performance measurement for one algorithm on one fingerprint."""
    records = _load("metrics")
    records.append({
        "fingerprint_id": fingerprint_id,
        "algorithm": algorithm,
        "execution_time_ms": execution_time_ms,
        "minutiae_count": minutiae_count,
        "accuracy": accuracy,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save("metrics", records)


def get_metrics_for_algorithm(algorithm):
    """Return all metrics recorded for a given algorithm."""
    return [m for m in _load("metrics") if m["algorithm"] == algorithm]