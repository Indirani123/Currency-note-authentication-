import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Upload both images
uploaded = files.upload()

# Load images from uploaded files
img1 = cv2.imdecode(np.frombuffer(uploaded['authentic_note.jpg'], np.uint8), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imdecode(np.frombuffer(uploaded['test_note.jpg'], np.uint8), cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort the matches based on distances
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the result in Colab
cv2_imshow(img_matches)

# Analyze the number of matches and distances to determine authenticity
if len(matches) > 50:  # You may tune this threshold
    print("Currency note is authentic.")
else:
    print("Currency note is counterfeit.")
