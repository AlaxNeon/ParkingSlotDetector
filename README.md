# Parking Slot Detection System  
### Developed for PNT-LAB Challenge – IIT Tirupati  

This project presents a lightweight and interpretable computer vision system to detect and classify parking slot occupancy using classical image processing techniques. Designed to work with both static images and video inputs, it provides a practical, non-AI-based alternative to deep learning models.  

---

## 🔍 Project Overview

The objective was to build a solution that could accurately identify and classify parking slots as **occupied** or **free** using just an image of the parking lot — without requiring any pretrained model or sensor data.  

To tackle this, the solution includes **two independent methods**, each implemented with its own executable and outputs:

---

## 🧪 Solution Approaches

### 1. Pixel Intensity Thresholding

This method processes a single parking lot image using:

- Grayscale conversion  
- Gaussian blur and adaptive thresholding  
- Morphological operations (median filtering, dilation)  
- Manual slot labeling using a custom slot picker  
- Non-zero pixel counting within each region to determine occupancy  

**Pros:**
- Lightweight, real-time friendly  
- No need for training or reference images  
- Highly explainable and customizable  

**Cons:**
- Sensitive to lighting, shadows, and noise  
- Requires precise threshold tuning  
- Fixed camera angle and manual annotation required  

---

### 2. Connected Component Analysis (CCA)

This approach compares a current parking image with a **manually created empty reference image** to detect foreground blobs (vehicles) by:

- Image differencing  
- Binary thresholding and morphological noise cleaning  
- Connected component labeling  
- Overlap analysis with manually labeled slots (from CSV)  

**Pros:**
- More robust to lighting variations  
- No hard thresholds required  
- Automatically detects diverse vehicle shapes and sizes  

**Cons:**
- Needs an accurate empty parking lot reference image  
- Requires pixel-perfect camera alignment  
- Slightly higher preprocessing time  

---

## 📁 Folder Descriptions

- `Solution/` – Contains the Python scripts and output folders for both methods and multiple other experiments.
- `Document/` – Includes the final project report with all explanations, figures, methodology, and performance evaluation.

---

## 📋 Key Features

- Works with both static images and adaptable to live video feeds  
- Outputs annotated images and CSV summaries  
- Pure Python + OpenCV — no AI model or GPU needed  
- Simple, interpretable logic ideal for deployment on edge devices  

---

## 📌 Why Not Deep Learning?

While models like YOLO or SSD are powerful, they were not used due to:

- No labeled datasets available  
- High training complexity and GPU requirements  
- Difficult deployment on lightweight or embedded devices  
- Need for transparent, explainable methods

This project shows that **classical techniques** still deliver practical, scalable solutions when designed thoughtfully.

---

## 🧠 Author

**Somnath Gorai**  
BCA IoT – Jain (Deemed-to-be) University  

---