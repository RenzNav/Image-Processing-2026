# Fingerprint Skeletonisation

Image Processing group project — Final Year Computer Science.

The aim of this project is to process fingerprint images and reduce their ridges to one-pixel-wide skeletons. We will implement and compare three skeletonisation algorithms before selecting the most suitable one for our final system.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Runs the program, allows image selection, calls the algorithms and displays results |
| `preprocessing.py` | Prepares fingerprint images for skeletonisation |
| `matching.py` | Contains the fingerprint matching function (not yet implemented) |
| `algorithms/zhang_suen.py` | Clarence's Zhang–Suen thinning implementation |
| `algorithms/medial_axis.py` | Justin's Medial Axis implementation |
| `algorithms/morphological.py` | Dami's Morphological skeletonisation implementation |
| `requirements.txt` | Lists the Python libraries required |
| `.gitignore` | Prevents local environment and temporary files from being tracked |

## Setup

Clone the repository and open the project folder in VS Code.

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

Run the program:

```powershell
python main.py
```

Or just run as normal (CTRL+ALT+N)

A file selection window will open. Select a fingerprint image in JPG or PNG format.

## How the Program Works

1. The user selects a fingerprint image.
2. `preprocessing.py` converts it to grayscale, resizes it, improves contrast, reduces noise and converts it to binary.
3. Very small disconnected white regions are removed using connected-component filtering.
4. The same binary image is passed to each skeletonisation algorithm.
5. `main.py` displays the original image, preprocessed image and available skeletonisation results.
6. Fingerprint matching will be integrated after the algorithms are implemented.

The current preprocessing configuration uses CLAHE, Gaussian blur, adaptive thresholding and connected-component filtering with a minimum area of 10 pixels. Morphological opening and closing are disabled because initial testing showed that they removed useful ridge detail.

## Implementing Your Algorithm
Keep the existing function names:

- `zhang_suen_thinning(binary_img)`
- `medial_axis_thinning(binary_img)`
- `morphological_thinning(binary_img)`

**Input:** A two-dimensional NumPy array containing a binary fingerprint. White ridges have pixel value `255`, and the black background has value `0`.

**Output:** A two-dimensional NumPy array of the same dimensions, containing the skeletonised fingerprint, ideally with ridges one pixel wide.

`main.py` automatically runs implemented algorithms and skips any that still raise `NotImplementedError`.

To test only your algorithm, you can temporarily comment out the other entries in the `ALGORITHMS` dictionary in `main.py`.
