import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_mapping(source, reference):
    """Match histogram of source image to reference image."""
    source_hist, bins = np.histogram(source.ravel(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.ravel(), 256, [0, 256])

    # Compute cumulative distribution function (CDF)
    source_cdf = np.cumsum(source_hist) / source_hist.sum()
    reference_cdf = np.cumsum(reference_hist) / reference_hist.sum()

    # Create a mapping from source to reference
    mapping = np.interp(source_cdf, reference_cdf, np.arange(256))

    # Apply mapping
    mapped = mapping[source.ravel()].reshape(source.shape).astype(np.uint8)
    return mapped

def histogram_equalization(image):
    """Perform histogram equalization."""
    return cv2.equalizeHist(image)

def contrast_stretching(image):
    """Perform contrast stretching (min-max normalization)."""
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Load images
source_img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

if source_img is None or reference_img is None:
    print("Error: Image not loaded. Check file paths.")
    exit()

# Apply transformations
mapped_img = histogram_mapping(source_img, reference_img)
equalized_img = histogram_equalization(source_img)
stretched_img = contrast_stretching(source_img)

# Display results
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

titles = ["Original", "Histogram Mapped", "Equalized", "Contrast Stretched"]
images = [source_img, mapped_img, equalized_img, stretched_img]

for i in range(4):
    axes[0, i].imshow(images[i], cmap="gray")
    axes[0, i].set_title(titles[i])
    axes[0, i].axis("off")

    # Plot histograms
    axes[1, i].hist(images[i].ravel(), bins=256, range=[0, 256], color='black', histtype='step')
    axes[1, i].set_title(f"Histogram - {titles[i]}")
    axes[1, i].set_xlim([0, 256])

plt.tight_layout()
plt.show()
