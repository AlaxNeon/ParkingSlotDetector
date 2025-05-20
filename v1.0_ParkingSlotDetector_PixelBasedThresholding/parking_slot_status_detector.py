import cv2
import pickle
import cvzone
import numpy as np
import os
import csv

# === PARAMETERS ===
val1 = 25  # blockSize (odd)
val2 = 16  # constant C
val3 = 5   # median blur kernel size (odd)
width, height = 68, 28  # Slot size
threshold_count = 270    # Threshold to decide free/occupied

# Ensure val1 and val3 are odd
if val1 % 2 == 0: val1 += 1
if val3 % 2 == 0: val3 += 1

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load image and position list
img = cv2.imread('images/parking_lot.png')  # Use forward slashes for cross-platform
with open('parking_slots', 'rb') as f:
    posList = pickle.load(f)

# === Image Processing ===
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, val1, val2)
imgThres = cv2.medianBlur(imgThres, val3)
kernel = np.ones((3, 3), np.uint8)
imgThres = cv2.dilate(imgThres, kernel, iterations=1)

# === Check Parking Slots ===
spaces = 0
for pos in posList:
    x, y = pos
    w, h = width, height

    imgCrop = imgThres[y:y + h, x:x + w]
    count = cv2.countNonZero(imgCrop)

    if count < threshold_count:
        color = (0, 200, 0)
        thickness = 5
        spaces += 1
    else:
        color = (0, 0, 200)
        thickness = 2

    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(img, str(count), (x, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# === Overlay Result Summary ===
total_slots = len(posList)
free_slots = spaces
occupied_slots = total_slots - free_slots

cvzone.putTextRect(img, f'Free: {free_slots}/{total_slots}', (40, 50), thickness=3, offset=10,
                   colorR=(0, 200, 0))

# === Save Output Image ===
output_img_path = os.path.join(output_dir, "result_parking_status.png")
cv2.imwrite(output_img_path, img)
print(f"Image saved to: {output_img_path}")

# === Save Output CSV ===
output_csv_path = os.path.join(output_dir, "parking_status.csv")
with open(output_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Total Slots", "Occupied Slots", "Free Slots"])
    writer.writerow([total_slots, occupied_slots, free_slots])

print(f"CSV saved to: {output_csv_path}")
