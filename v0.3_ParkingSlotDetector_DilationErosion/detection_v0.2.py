import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

# === CONFIG ===
EMPTY_IMAGE_PATH = "images/empty_parking_lot.png"
CURRENT_IMAGE_PATH = "images/parking_lot.png"
ANNOTATIONS_PATH = "annotations/empty_parking_lot_coco_polygon.json"
MASK_OUTPUT_DIR = "masks"
YOLO_MODEL_PATH = "models/yolov8-seg.pt"

# IOU thresholds for classification
IOU_THRESHOLD_OCCUPIED = 0.3
IOU_THRESHOLD_PARTIAL = 0.15

os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# === STEP 1: Load COCO Annotations ===
def load_parking_slot_polygons(annotation_path, image_path):
    with open(annotation_path) as f:
        coco = json.load(f)

    img_name = os.path.basename(image_path)
    image_info = next((img for img in coco['images'] if img['file_name'] == img_name), None)
    if not image_info:
        raise ValueError("Image not found in annotations.")

    img_id = image_info['id']
    polygons = []
    for ann in coco['annotations']:
        if ann['image_id'] == img_id:
            polygons.append(np.array(ann['segmentation'][0]).reshape((-1, 2)).astype(np.int32))

    return polygons

# === STEP 2: Generate Binary Masks for Parking Slots ===
def generate_masks(empty_image, polygons):
    h, w = empty_image.shape[:2]
    masks = []
    for idx, poly in enumerate(polygons):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        masks.append(mask)
        cv2.imwrite(f"{MASK_OUTPUT_DIR}/slot_{idx}.png", mask)
    return masks

# === STEP 3: Run YOLOv8 Segmentation ===
def get_vehicle_masks(model_path, image_path):
    model = YOLO(model_path)
    results = model(image_path)[0]
    masks = []

    if results.masks is not None:
        for seg in results.masks.data:
            mask = seg.cpu().numpy().astype(np.uint8) * 255
            masks.append(mask)
    return masks

# === STEP 4: Check Occupancy with Visibility & IOU thresholds ===
def check_occupancy(slot_masks, car_masks):
    statuses = []
    for idx, slot in enumerate(slot_masks):
        max_iou = 0
        for car in car_masks:
            # Resize car mask to slot mask size if different
            if car.shape != slot.shape:
                car_resized = cv2.resize(car, (slot.shape[1], slot.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                car_resized = car

            slot_bin = (slot > 0).astype(np.uint8) * 255
            car_bin = (car_resized > 0).astype(np.uint8) * 255

            intersection = cv2.bitwise_and(slot_bin, car_bin)
            union = cv2.bitwise_or(slot_bin, car_bin)

            iou = np.sum(intersection > 0) / (np.sum(union > 0) + 1e-6)
            max_iou = max(max_iou, iou)

        # Decide occupancy status based on thresholds
        if max_iou >= IOU_THRESHOLD_OCCUPIED:
            status = "Occupied"
        elif max_iou >= IOU_THRESHOLD_PARTIAL:
            status = "Partially Occupied"
        else:
            status = "Empty"

        print(f"Slot {idx} max IOU: {max_iou:.3f} --> {status}")
        statuses.append(status)
    return statuses

# === STEP 5: Visualize Results with color coding ===
def visualize_results(base_image_path, slot_polygons, statuses):
    image = cv2.imread(base_image_path)
    for i, poly in enumerate(slot_polygons):
        if statuses[i] == "Occupied":
            color = (0, 0, 255)  # Red
        elif statuses[i] == "Partially Occupied":
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green

        cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(image, statuses[i], tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite("output/parking_result_v0.2.png", image)
    print("✅ Output saved to output/parking_result.png")

# === MAIN ===
if __name__ == "__main__":
    empty_img = cv2.imread(EMPTY_IMAGE_PATH)
    slot_polygons = load_parking_slot_polygons(ANNOTATIONS_PATH, EMPTY_IMAGE_PATH)
    slot_masks = generate_masks(empty_img, slot_polygons)
    vehicle_masks = get_vehicle_masks(YOLO_MODEL_PATH, CURRENT_IMAGE_PATH)
    occupancy_statuses = check_occupancy(slot_masks, vehicle_masks)
    visualize_results(CURRENT_IMAGE_PATH, slot_polygons, occupancy_statuses)

    # === STEP 6: Save Occupancy Stats to CSV ===
    import csv

    # Count categories
    total_slots = len(occupancy_statuses)
    occupied = occupancy_statuses.count("Occupied")
    partial = occupancy_statuses.count("Partially Occupied")
    empty = occupancy_statuses.count("Empty")

    csv_path = os.path.join("output", "occupancy_stats_v0.2.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Slot_Index", "Status"])
        for idx, status in enumerate(occupancy_statuses):
            writer.writerow([idx, status])

        writer.writerow([])  # Blank line
        writer.writerow(["Total_Slots", "Occupied", "Partially_Occupied", "Empty"])
        writer.writerow([total_slots, occupied, partial, empty])

    print(f"📄 CSV saved to {csv_path}")
