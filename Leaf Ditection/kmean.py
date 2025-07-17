import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load and resize image ---
img = cv2.imread("6.JPG")  # Replace with your image
img = cv2.resize(img, (512, 512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Convert to HSV and apply Gaussian Blur ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

# --- Smart background removal using KMeans segmentation ---
Z = img.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

segmented = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
seg_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
_, background_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
leaf_mask = cv2.bitwise_not(background_mask)

# --- Morphological cleanup ---
kernel = np.ones((5, 5), np.uint8)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

# --- Dominant hue calculation ---
hue_channel = hsv_blurred[:, :, 0]
masked_hue = hue_channel[leaf_mask == 255]
dominant_hue = int(np.bincount(masked_hue).argmax())

lower_dominant = np.array([max(dominant_hue - 10, 0), 40, 40])
upper_dominant = np.array([min(dominant_hue + 10, 180), 255, 255])

# --- Healthy mask based on dominant hue ---
healthy_mask = cv2.inRange(hsv_blurred, lower_dominant, upper_dominant)
healthy_mask = cv2.bitwise_and(healthy_mask, leaf_mask)
healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_CLOSE, kernel)

# --- Disease mask ---
disease_mask = cv2.subtract(leaf_mask, healthy_mask)
disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

# --- Contour & area calculation ---
contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

leaf_area = cv2.countNonZero(leaf_mask)
disease_area = cv2.countNonZero(disease_mask)
healthy_area = cv2.countNonZero(healthy_mask)
disease_percent = (disease_area / leaf_area * 100) if leaf_area else 0
healthy_percent = (healthy_area / leaf_area * 100) if leaf_area else 0

# --- Visualization ---
leaf_extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=leaf_mask)
disease_visual = cv2.bitwise_and(img_rgb, img_rgb, mask=disease_mask)

titles = [
    "Original", "KMeans Mask", "Leaf Mask",
    "Healthy Mask (Dominant Hue)", "Disease Mask",
    "Extracted Leaf", "Disease Area", "Contour Overlay"
]
images = [
    img_rgb, background_mask, leaf_mask,
    healthy_mask, disease_mask,
    leaf_extracted, disease_visual, contour_img
]

plt.figure(figsize=(16, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i] if i not in [1, 2, 3, 4] else images[i], cmap='gray' if i in [1, 2, 3, 4] else None)
    plt.axis("off")

plt.subplot(3, 3, 9)
plt.axis("off")
text = f"Disease %: {disease_percent:.2f}%\nHealthy %: {healthy_percent:.2f}%"
plt.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
