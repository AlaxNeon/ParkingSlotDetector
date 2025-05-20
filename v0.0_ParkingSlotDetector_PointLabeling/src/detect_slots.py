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
    contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return cv2.pointPolygonTest(contour, (px, py), False) >= 0

# Output result summary
results = []

# Process each image
for img_name in os.listdir("images"):
    img_path = os.path.join("images", img_name)
    image = cv2.imread(img_path)

    detections = model.predict(img_path, conf=0.3)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]

    image_result = {
        "image": img_name,
        "total_slots": 0,
        "occupied": 0,
        "free": 0
    }

    # Process each slot point
    for idx, slot in enumerate(slot_points.get(img_name, [])):
        px, py = slot['x'], slot['y']
        slot_id = slot['slot_id']

        # Extract only the numeric part from slot_id (e.g., parking_slot_3 → 3)
        slot_number = ''.join(filter(str.isdigit, slot_id))

        occupied = any(is_point_inside_bbox(px, py, box) for box in boxes)

        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.circle(image, (px, py), 6, color, -1)
        cv2.putText(image, str(slot_number), (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if occupied:
            image_result["occupied"] += 1
        else:
            image_result["free"] += 1

        image_result["total_slots"] += 1

    # Save the result image
    cv2.imwrite(f"output/{img_name}", image)
    results.append(image_result)

# Save CSV summary
with open("output/results.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image", "total_slots", "occupied", "free"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Detection complete using polygon test. Results saved in /output.")