import cv2
import numpy as np
import pandas as pd
import os
import cvzone
import csv

# === PARAMETERS ===
THRESHOLD_BINARY_DIFF = 25
MIN_BLOB_AREA = 300     # Ignore small objects (e.g., shadows, litter)

# === Load Images ===
empty_img = cv2.imread("images/parking_lot_empty.png")
current_img = cv2.imread("images/parking_lot.png")

if empty_img is None or current_img is None:
    raise FileNotFoundError("One or both images not found in input folder!")

# === Resize images to match if needed ===
if empty_img.shape != current_img.shape:
    current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))

# === Convert to Grayscale ===
gray_empty = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/gray_empty.png", gray_empty)
cv2.imwrite("output/gray_current.png", gray_current)

# === Difference & Threshold ===
diff_img = cv2.absdiff(gray_empty, gray_current)
cv2.imwrite("output/difference_image.png", diff_img)

# === Binary Thresholding ===
_, fg_mask = cv2.threshold(diff_img, THRESHOLD_BINARY_DIFF, 255, cv2.THRESH_BINARY)
cv2.imwrite("output/foreground_mask_raw.png", fg_mask)

# === Morphological Cleaning ===
kernel = np.ones((3, 3), np.uint8)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imwrite("output/foreground_mask_cleaned.png", fg_mask)

# === Connected Component Analysis ===
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)

vehicle_blobs = []  # (bounding_box, area)
for i in range(1, num_labels):  # Skip background
    x, y, w, h, area = stats[i]
    if area >= MIN_BLOB_AREA:
        vehicle_blobs.append(((x, y, w, h), area))

# === Visualize Blobs ===
blob_img = current_img.copy()
for (x, y, w, h), area in vehicle_blobs:
    cv2.rectangle(blob_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(blob_img, f"Blob Area: {area}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

# === Save Intermediate Blob Output ===
cv2.imwrite("output/foreground_mask_cleaned.png", fg_mask)
cv2.imwrite("output/blob_detection.png", blob_img)

# === Final Detection Using CSV Slot Map ===
slot_data = pd.read_csv("slots_data\slots.csv")
final_img = current_img.copy()
occupied_slots, free_slots = [], []

for _, row in slot_data.iterrows():
    sid, x, y, w, h = int(row['id']), int(row['x']), int(row['y']), int(row['w']), int(row['h'])
    slot_box = (x, y, x + w, y + h)
    is_occupied = False

    for (vx, vy, vw, vh), _ in vehicle_blobs:
        vehicle_box = (vx, vy, vx + vw, vy + vh)

        # Calculate overlap
        x_overlap = max(0, min(slot_box[2], vehicle_box[2]) - max(slot_box[0], vehicle_box[0]))
        y_overlap = max(0, min(slot_box[3], vehicle_box[3]) - max(slot_box[1], vehicle_box[1]))
        overlap_area = x_overlap * y_overlap

        if overlap_area > 0:
            is_occupied = True
            break

    # Draw & label slot
    color = (0, 0, 255) if is_occupied else (0, 255, 0)
    cv2.rectangle(final_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(final_img, f"Slot {sid}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    if is_occupied:
        occupied_slots.append(sid)
    else:
        free_slots.append(sid)

# === Count Summary ===
total_slots = len(occupied_slots) + len(free_slots)
occupied_count = len(occupied_slots)
free_count = len(free_slots)

# === Overlay Summary Text ===
summary_text = f"Free: {free_count}/{total_slots}"
cvzone.putTextRect(final_img, summary_text, (40, 50), thickness=3, offset=10, colorR=(0, 200, 0))

# === Save Final Annotated Image ===
cv2.imwrite("output/parking_slot_status.png", final_img)

# === Count Summary ===
total_slots = len(occupied_slots) + len(free_slots)
occupied_count = len(occupied_slots)
free_count = len(free_slots)

# === Print Summary ===
print("\n===== PARKING STATUS =====")
print(f"Total:    {total_slots}")
print(f"Occupied: {occupied_count}")
print(f"Free:     {free_count}")
print("Final status saved to: output/parking_slot_status.png")

# === Save Summary to CSV ===
output_csv_path = os.path.join("output", "parking_slot_status.csv")
with open(output_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Total Slots", "Occupied Slots", "Free Slots"])
    writer.writerow([total_slots, occupied_count, free_count])

print(f"CSV saved to: {output_csv_path}")