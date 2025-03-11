import cv2
import numpy as np

# Load images
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB Feature Detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches (for debugging)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Feature Matches", match_img)
cv2.imwrite("feature_matches.jpg", match_img)

# Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute Homography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp img1 to align with img2
height, width, channels = img2.shape
result = cv2.warpPerspective(img1, H, (width * 2, height))
result[0:height, 0:width] = img2  # Overlay img2 onto the result

# Show and save the panorama
cv2.imshow("Panorama", result)
cv2.imwrite("panorama_result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
