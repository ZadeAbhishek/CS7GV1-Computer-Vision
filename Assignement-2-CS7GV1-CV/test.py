import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Define Camera Parameters
focal_length = 4152.073
baseline = 176.252

# Load Stereo Images
left_image = cv2.imread('./Images/im0.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('./Images/im1.png', cv2.IMREAD_GRAYSCALE)

if left_image is None or right_image is None:
    raise ValueError("Images not found! Make sure the images are in the correct directory.")

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(left_image, cmap='gray'), plt.title("Left Image")
plt.subplot(1, 2, 2), plt.imshow(right_image, cmap='gray'), plt.title("Right Image")
plt.show()

# Step 1: Register the images using cross-correlation
def register_images(left_image, right_image):
    correlation = correlate2d(left_image, right_image, boundary='wrap', mode='same')
    y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    y_shift -= left_image.shape[0] // 2
    x_shift -= left_image.shape[1] // 2
    print(f"Detected shift: x = {x_shift}, y = {y_shift}")
    
    # Apply translation to align right_image to left_image
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    aligned_right_image = cv2.warpAffine(right_image, translation_matrix, (right_image.shape[1], right_image.shape[0]))
    return aligned_right_image

right_image_aligned = register_images(left_image, right_image)

# Display the aligned images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(left_image, cmap='gray'), plt.title("Left Image")
plt.subplot(1, 2, 2), plt.imshow(right_image_aligned, cmap='gray'), plt.title("Aligned Right Image")
plt.show()

# Step 2: Feature Detection and Matching with Enhanced Parameters
def detect_and_match_features(left_image, right_image, ratio_test_threshold=0.6):
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_image, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)
            
    return good_matches, keypoints_left, keypoints_right

good_matches, keypoints_left, keypoints_right = detect_and_match_features(left_image, right_image_aligned)

# Step 3: Calculate Disparity and Depth
def calculate_disparity_and_depth(keypoints_left, keypoints_right, good_matches, focal_length, baseline, doffs=0):
    disparities = []
    xl_coords = []
    yl_coords = []
    
    for match in good_matches:
        x_left, y_left = keypoints_left[match.queryIdx].pt
        x_right, _ = keypoints_right[match.trainIdx].pt
        disparity = (x_left - x_right) + doffs

        disparities.append(disparity)
        xl_coords.append(x_left)
        yl_coords.append(y_left)
    
    disparity_map = np.zeros(left_image.shape, dtype=np.float32)
    for disparity, x_left, y_left in zip(disparities, xl_coords, yl_coords):
        disparity_map[int(y_left), int(x_left)] = disparity
    
    disparity_map[disparity_map == 0] = 0.1  # Avoid division by zero
    depth_map = (focal_length * baseline) / disparity_map
    return disparity_map, depth_map

disparity_map, depth_map = calculate_disparity_and_depth(
    keypoints_left, keypoints_right, good_matches, focal_length, baseline)

# Display Disparity Map
plt.figure(figsize=(10, 5))
plt.imshow(disparity_map, cmap='jet')
plt.colorbar()
plt.title("Disparity Map after Registration and Improved Matching")
plt.show()

# Display Depth Map
plt.figure(figsize=(10, 5))
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.title("Depth Map")
plt.show()