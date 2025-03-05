import cv2
import numpy as np
import matplotlib.pyplot as plt

def smooth_image(image):
    """Apply Gaussian Blur for smoothing."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen_image(image):
    """Apply a sharpening filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Load image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not loaded. Check file path.")
    exit()

# Apply smoothing and sharpening
smoothed = smooth_image(image)
sharpened = sharpen_image(image)

# Display results
titles = ["Original", "Smoothed", "Sharpened"]
images = [image, smoothed, sharpened]

plt.figure(figsize=(10, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()
