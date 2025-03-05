import cv2
import numpy as np
import matplotlib.pyplot as plt

def scale_image(image, scale_x, scale_y):
    """Resize image with given scale factors."""
    return cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

def rotate_image(image, angle):
    """Rotate image by given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def translate_image(image, tx, ty):
    """Translate image by given tx, ty pixels."""
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

def shear_image(image, shear_x, shear_y):
    """Apply shear transformation."""
    (h, w) = image.shape[:2]
    matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h))

# Load image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not loaded. Check file path.")
    exit()

# Apply transformations
scaled = scale_image(image, 1.5, 1.5)  # 1.5x scaling
rotated = rotate_image(image, 30)  # Rotate by 30 degrees
translated = translate_image(image, 50, 30)  # Shift right 50px, down 30px
sheared = shear_image(image, 0.2, 0.1)  # Shear transformation

# Display results
titles = ["Original", "Scaled", "Rotated", "Translated", "Sheared"]
images = [image, scaled, rotated, translated, sheared]

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()
