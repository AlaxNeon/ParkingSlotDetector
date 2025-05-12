# ParkingSlotDetector

This is an OpenCV-based project to detect the occupancy status of parking slots in a parking place using YOLOv8 and point-based annotations.

## Steps to Get the Project Running

### Step 1:  
Install the required packages:  
```
pip install ultralytics opencv-python numpy
```

### Step 2:  
Label the parking slots:  
1. Go to https://www.makesense.ai  
2. Upload your parking lot image(s)  
3. Use the Point annotation tool to mark the center of each parking slot  
4. Export the annotations in CSV format

### Step 3:  
Organize your project directory as follows:
```
ParkingSlotDetector_PointLabeling/
+-- images/                  # Contains input parking lot image(s)
¦   +-- parking_lot.png
+-- annotations/             # Contains the exported CSV annotation file
¦   +-- slot_points.csv
+-- models/                  # Contains your YOLOv8 model
¦   +-- yolov8_model.pt
+-- output/                  # Will contain the output images and results
+-- src/
¦   +-- detect_slots.py      # Main detection script
¦   +-- utils.py             # Utility to load annotations
```

### Step 4:  
Run the detection script:
```
python src/detect_slots.py
```

The script will:
- Detect vehicles using YOLOv8
- Check which parking slot points are inside vehicle bounding boxes
- Mark occupied (red) and free (green) slots
- Save the annotated image and result CSV in the output/ folder
