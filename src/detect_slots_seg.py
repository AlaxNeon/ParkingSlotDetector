from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from utils import load_slot_points  # CSV: slot_id, x, y, filename

# Load segmentation model
model = YOLO("models/yolov8n-seg.pt")  # or yolov8s-seg.pt for better accuracy

# Load slot annotations
slot_points = load_slot_points("annotations/slot_points.csv")

results = []

# Loop through images
for img_name in os.listdir("images"):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join("images", img_name)
    image = cv2.imread(img_path)
    orig_h, orig_w = image.shape[:2]

    # Run segmentation with low threshold
    detect_result = model.predict(img_path, conf=0.25, task="segment")[0]
    masks = detect_result.masks

    image_result = {
        "image": img_name,
        "total_slots": 0,
        "occupied": 0,
        "free": 0
    }

    if masks is not None:
        mask_array = masks.data.cpu().numpy()  # shape: (N, H, W)
        mask_h, mask_w = mask_array.shape[1:]

        scale_x = mask_w / orig_w
        scale_y = mask_h / orig_h

        for slot in slot_points.get(img_name, []):
            px, py = int(slot["x"]), int(slot["y"])
            slot_id = slot["slot_id"]
            slot_number = ''.join(filter(str.isdigit, slot_id))

            mx = int(px * scale_x)
            my = int(py * scale_y)

            if 0 <= my < mask_h and 0 <= mx < mask_w:
                occupied = any(mask[my, mx] > 0 for mask in mask_array)
            else:
                occupied = False

            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.circle(image, (px, py), 6, color, -1)
            cv2.putText(image, str(slot_number), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if occupied:
                image_result["occupied"] += 1
            else:
                image_result["free"] += 1

            image_result["total_slots"] += 1
    else:
        print(f"⚠️ No masks found in {img_name}")

    # Save image
    cv2.imwrite(f"output/{img_name}", image)
    results.append(image_result)

# Save CSV
with open("output/results_seg.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "total_slots", "occupied", "free"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Segmentation-based parking slot detection complete (no class filter).")
