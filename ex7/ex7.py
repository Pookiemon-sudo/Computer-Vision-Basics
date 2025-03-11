import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = 'image.jpg'  # Provide your image path here
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Apply GaussianBlur to reduce noise (important for edge detection)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Step 3: Use Canny edge detector
edges = cv2.Canny(blurred_image, 100, 200)

# Step 4: Display the original and edge-detected images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Edge detected image
plt.subplot(1, 2, 2)
plt.title("Edge Detected Image")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
