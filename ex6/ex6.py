import cv2 as cv
import numpy as np

# Load image with error handling
im = cv.imread('image.jpg')
if im is None:
    print("Error: Image not found or path incorrect")
    exit()

# Resize image
img = cv.resize(im, (480, 360))

# Display original image
cv.imshow('Real Image', img)
cv.waitKey(0)

def apply_morphology(operation_name, img, morph_type):
    """General function to apply morphological transformations"""
    ker = np.ones((5, 5), np.uint8)
    result = cv.morphologyEx(img, morph_type, ker) if morph_type else cv.erode(img, ker, iterations=1)
    cv.imshow(operation_name, result)
    cv.waitKey(0)

# Apply transformations
apply_morphology("Erosion", img, None)  # Erosion
apply_morphology("Dilation", img, cv.MORPH_DILATE)  # Dilation
apply_morphology("Opening", img, cv.MORPH_OPEN)  # Opening
apply_morphology("Closing", img, cv.MORPH_CLOSE)  # Closing

cv.destroyAllWindows()
