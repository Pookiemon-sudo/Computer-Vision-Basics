import cv2
import numpy as np

# Load 24-bit color image
img = cv2.imread("image.jpg")

# Convert to 8-bit grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("8bit_image.png", gray)  # Save as 8-bit PNG

# Convert to 4-bit (16 shades of gray)
gray_4bit = (gray // 16) * 16  # Reduce to 16 shades
cv2.imwrite("4bit_image.png", gray_4bit)  # PNG still stores it as 8-bit

# Convert to 1-bit (black & white)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite("1bit_image.png", binary)  # Saved as PNG, but still 8-bit

print("Conversion completed: 8-bit, 4-bit, and 1-bit images saved as PNG.")
