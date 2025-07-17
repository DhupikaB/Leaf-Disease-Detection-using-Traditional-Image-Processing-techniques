import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load and resize image ---
img = cv2.imread("healthy4.JPG")
img = cv2.resize(img, (512, 512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Convert to HSV and apply Gaussian Blur ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

# --- Background removal using gray mask ---
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 60, 255])
background_mask = cv2.inRange(hsv_blurred, lower_gray, upper_gray)
leaf_mask = cv2.bitwise_not(background_mask)

# --- Morphological cleanup ---
kernel = np.ones((5, 5), np.uint8)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

# --- Dominant healthy color detection using Mode Hue ---
hue_channel = hsv_blurred[:, :, 0]
masked_hue = hue_channel[leaf_mask == 255]
dominant_hue = int(np.bincount(masked_hue).argmax())

# Create dynamic healthy range around dominant hue
lower_dominant = np.array([max(dominant_hue - 10, 0), 40, 40])
upper_dominant = np.array([min(dominant_hue + 10, 180), 255, 255])

# --- Healthy mask based on dominant color ---
healthy_mask = cv2.inRange(hsv_blurred, lower_dominant, upper_dominant)
healthy_mask = cv2.bitwise_and(healthy_mask, leaf_mask)
healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_CLOSE, kernel)

# --- Subtract healthy from leaf to get disease mask ---
disease_mask = cv2.subtract(leaf_mask, healthy_mask)
disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

# --- Contour detection for diseased areas ---
contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 3)

# --- Area calculations ---
leaf_area = cv2.countNonZero(leaf_mask)
disease_area = cv2.countNonZero(disease_mask)
healthy_area = cv2.countNonZero(healthy_mask)

disease_percentage = (disease_area / leaf_area) * 100 if leaf_area else 0
healthy_percentage = (healthy_area / leaf_area) * 100 if leaf_area else 0

# --- Extracted visualizations ---
leaf_extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=leaf_mask)
disease_visual = cv2.bitwise_and(img_rgb, img_rgb, mask=disease_mask)

# --- Plotting without loops ---
plt.figure(figsize=(16, 10))

plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Background Mask")
plt.imshow(background_mask, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Leaf Mask")
plt.imshow(leaf_mask, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Healthy Mask (Dominant Hue)")
plt.imshow(healthy_mask, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("Disease Mask")
plt.imshow(disease_mask, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("Extracted Leaf")
plt.imshow(leaf_extracted)
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("Disease Area")
plt.imshow(disease_visual)
plt.axis("off")

plt.subplot(3, 3, 8)
plt.title("Contour Overlay")
plt.imshow(contour_img)
plt.axis("off")

plt.subplot(3, 3, 9)
plt.axis("off")
text = f"Disease %: {disease_percentage:.2f}%\nHealthy %: {healthy_percentage:.2f}%"
plt.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
