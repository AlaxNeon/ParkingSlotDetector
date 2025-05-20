from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from utils import load_slot_points

# Load YOLOv8 segmentation model
model = YOLO("models/yolov8s-seg.pt")

# Load annotated slot points with visibility tags
slot_points = load_slot_points("annotations/slot_points.csv")

results = []

for img_name in os.listdir("images"):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join("images", img_name)
    image = cv2.imread(img_path)
    orig_h, orig_w = image.shape[:2]

    # Run segmentation prediction
    prediction = model.predict(img_path, conf=0.15, task="segment")[0]
    masks = prediction.masks
    boxes = prediction.boxes.xyxy.cpu().numpy() if prediction.boxes is not None else []

    image_result = {
        "image": img_name,
        "total_slots": 0,
        "occupied": 0,
        "free": 0,
        "possibly_occupied": 0,
        "unknown": 0
    }

    debug_image = image.copy()

    if masks is not None and masks.data is not None:
        mask_array = masks.data.cpu().numpy()
        mask_h, mask_w = mask_array.shape[1:]

        scale_x = mask_w / orig_w
        scale_y = mask_h / orig_h

        # Overlay segmentation masks for visual debugging
        mask_overlay = np.zeros_like(image)
        for mask in mask_array:
            binary_mask = (mask * 255).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (orig_w, orig_h))
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, (0, 255, 255), thickness=cv2.FILLED)

        debug_image = cv2.addWeighted(debug_image, 0.7, mask_overlay, 0.3, 0)

        # Process all slots
        for slot in slot_points.get(img_name, []):
            px, py = int(slot["x"]), int(slot["y"])
            visibility = slot["visibility"].strip().lower()
            slot_id = slot["slot_id"]
            slot_number = ''.join(filter(str.isdigit, slot_id))

            mx = int(px * scale_x)
            my = int(py * scale_y)

            # Use a slightly larger patch
            patch_size = 5  # This gives an 11x11 patch
            occupied = False

            if 0 <= my < mask_h and 0 <= mx < mask_w:
                for mask in mask_array:
                    y1 = max(0, my - patch_size)
                    y2 = min(mask_h, my + patch_size + 1)
                    x1 = max(0, mx - patch_size)
                    x2 = min(mask_w, mx + patch_size + 1)
                    if np.any(mask[y1:y2, x1:x2] > 0):
                        occupied = True
                        break

            # OPTIONAL: fallback using bounding boxes if mask didn't catch it
            if not occupied and boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        occupied = True
                        break

            # Visibility-aware logic
            if visibility == "fully_visible":
                status = "Occupied" if occupied else "Free"
            else:
                status = "Possibly Occupied" if occupied else "Unknown"

            # Display
            color_map = {
                "Occupied": (0, 0, 255),
                "Free": (0, 255, 0),
                "Possibly Occupied": (0, 255, 255),
                "Unknown": (128, 128, 128)
            }
            color = color_map[status]

            cv2.circle(debug_image, (px, py), 6, color, -1)
            cv2.putText(debug_image, slot_number, (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            image_result["total_slots"] += 1
            image_result[status.lower().replace(" ", "_")] += 1

    else:
        print(f"⚠️ No segmentation masks found in {img_name}")
        # Optional: fallback to bounding box only if masks are missing
        for slot in slot_points.get(img_name, []):
            px, py = int(slot["x"]), int(slot["y"])
            visibility = slot["visibility"].strip().lower()
            slot_id = slot["slot_id"]
            slot_number = ''.join(filter(str.isdigit, slot_id))

            occupied = False
            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= px <= x2 and y1 <= py <= y2:
                    occupied = True
                    break

            if visibility == "fully_visible":
                status = "Occupied" if occupied else "Free"
            else:
                status = "Possibly Occupied" if occupied else "Unknown"

            color_map = {
                "Occupied": (0, 0, 255),
                "Free": (0, 255, 0),
                "Possibly Occupied": (0, 255, 255),
                "Unknown": (128, 128, 128)
            }
            color = color_map[status]

            cv2.circle(debug_image, (px, py), 6, color, -1)
            cv2.putText(debug_image, slot_number, (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            image_result["total_slots"] += 1
            image_result[status.lower().replace(" ", "_")] += 1

    # Save the debug image
    cv2.imwrite(f"output/{img_name}", debug_image)
    results.append(image_result)

# Save results CSV
with open("output/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "total_slots", "occupied", "free", "possibly_occupied", "unknown"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Detection complete with segmentation, patch logic, and fallback bounding box check.")
