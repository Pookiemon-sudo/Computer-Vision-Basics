import cv2
import numpy as np

# Load the image
image = cv2.imread("image.jpg")  # Change to your image file

# Define color space conversion functions
def convert_color_space(image, conversion_code):
    return cv2.cvtColor(image, conversion_code)

# List of available color space conversions
color_spaces = {
    "GRAY": cv2.COLOR_BGR2GRAY,
    "HSV": cv2.COLOR_BGR2HSV,
    "LAB": cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "XYZ": cv2.COLOR_BGR2XYZ,
    "LUV": cv2.COLOR_BGR2LUV
}

# Convert to all color spaces and save results
for name, code in color_spaces.items():
    converted_img = convert_color_space(image, code)
    cv2.imwrite(f"{name}_image.png", converted_img)

print("Color space conversions completed and saved!")
