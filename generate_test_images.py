
import cv2
import numpy as np
from pathlib import Path

# Input and output folders
# Original fingerprint used for transformations
INPUT_PATH = Path("fingerprint_samples/Fingerprint_A.1.jpg")

# Save all transformed images alongside the existing samples
OUTPUT_DIR = Path("fingerprint_samples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the fingerprint in grayscale
image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Cannot load {INPUT_PATH}")

height, width = image.shape


# Save a transformed fingerprint
def save_image(name, result):
    cv2.imwrite(str(OUTPUT_DIR / name), result)
    print(f"Created: {name}")


# 1. Gaussian blur (3 images)
for size in [3, 5, 9]:
    blurred = cv2.GaussianBlur(image, (size, size), 0)
    save_image(f"blur_{size}x{size}.png", blurred)


# 2. Gaussian noise (3 images)
rng = np.random.default_rng(42)

for sigma in [10, 20, 35]:
    noise = rng.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    save_image(f"noise_{sigma}.png", noisy.astype(np.uint8))


# 3. Reduced contrast (3 images)
for contrast in [0.75, 0.50, 0.25]:
    adjusted = 127 + contrast * (image.astype(np.float32) - 127)
    adjusted = np.clip(adjusted, 0, 255)
    save_image(f"contrast_{contrast:.2f}.png", adjusted.astype(np.uint8))


# 4. Partial fingerprints (3 images)
# Retain a central square region and mask the rest
for fraction in [0.75, 0.50, 0.25]:
    side_fraction = np.sqrt(fraction)

    crop_width = int(width * side_fraction)
    crop_height = int(height * side_fraction)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    partial = np.full_like(image, 255)
    partial[y1:y1 + crop_height, x1:x1 + crop_width] = (
        image[y1:y1 + crop_height, x1:x1 + crop_width]
    )

    save_image(f"partial_{int(fraction * 100)}.png", partial)


# 5. Rotation (3 images)
for angle in [10, 20, 45]:
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )

    rotated = cv2.warpAffine(
        image, matrix, (width, height),
        borderValue=255
    )

    save_image(f"rotation_{angle}.png", rotated)


print("\nSuccessfully generated fingerprint test images!")
