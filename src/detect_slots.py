from ultralytics import YOLO
import cv2
import numpy as np
from utils import load_slot_points
import os
import csv

# Load YOLOv8 model
model = YOLO("models/yolov8_model.pt")  # Pre-trained or fine-tuned

# Load parking slot point annotations
slot_points = load_slot_points("annotations/slot_points.csv")

# Accurate check using OpenCV polygon containment
def is_point_inside_bbox(px, py, box):
    x1, y1, x2, y2 = map(int, box)
    # Define bounding box as a rectangle polygon
    contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    # Check if point lies inside polygon
    return cv2.pointPolygonTest(contour, (px, py), False) >= 0

# Output result summary
results = []

# Process each image
for img_name in os.listdir("images"):
    img_path = os.path.join("images", img_name)
    image = cv2.imread(img_path)

    # Predict vehicles using YOLOv8
    detections = model.predict(img_path, conf=0.3)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]

    image_result = {
        "image": img_name,
        "total_slots": 0,
        "occupied": 0,
        "free": 0
    }

    # Draw YOLO box centers (debugging)
    for box in boxes:
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)  # Blue dot = center

    # Check each slot point
    for idx, slot in enumerate(slot_points.get(img_name, [])):
        px, py = slot['x'], slot['y']
        slot_id = slot['slot_id']

        # Check if point is inside any vehicle bounding box
        occupied = any(is_point_inside_bbox(px, py, box) for box in boxes)

        # Draw circle on slot point
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.circle(image, (px, py), 6, color, -1)
        cv2.putText(image, str(slot_id), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Update counts
        if occupied:
            image_result["occupied"] += 1
        else:
            image_result["free"] += 1

        image_result["total_slots"] += 1

    # Save output image and record
    cv2.imwrite(f"output/{img_name}", image)
    results.append(image_result)

# Save final result summary to CSV
with open("output/results.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image", "total_slots", "occupied", "free"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Detection complete using polygon test. Results saved in /output.")
