import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Function to extract LBP features
def extract_lbp_features(image):
    # Resize image to a fixed size
    resized_image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Extract LBP features (uniform LBP)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    return lbp

# Process images in a folder and extract LBP features
def process_image_folder_lbp(image_folder_path):
    lbp_images = []
    
    # Read all image files from the folder
    for filename in os.listdir(image_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read image
            image_path = os.path.join(image_folder_path, filename)
            image = cv2.imread(image_path)
            
            # Extract LBP features
            lbp_image = extract_lbp_features(image)
            
            # Store the LBP images for later visualization
            lbp_images.append(lbp_image)
    
    return lbp_images

# Example usage
image_folder_path = 'animal_faces'  # Provide the path to your folder containing animal face images
lbp_images = process_image_folder_lbp(image_folder_path)

# Display the first LBP image
plt.imshow(lbp_images[0], cmap='gray')
plt.title("LBP Image")
plt.axis('off')
plt.show()
