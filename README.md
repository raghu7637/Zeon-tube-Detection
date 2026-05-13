# 🧪 Tube Detection and Pose Estimation

This repository contains my solution for the Zeon Systems technical challenge. The goal is to detect the center positions and rotation angles of microcentrifuge tubes across various surfaces and lighting conditions using a dataset of 70 overhead RGB images.

## 🧠 Technical Approach
I framed this as a Deep Learning computer vision problem. Given the transparency of the tubes and the varying backgrounds (white, black, mixed desks), traditional OpenCV techniques are too fragile. 

I utilized **YOLOv8** (Oriented Bounding Box / Pose Estimation) to robustly identify the tube lids and calculate their exact orientation angles. 

## 📊 Results & Metrics
* **Precision:** [Will update once model finishes training]
* **Recall:** [Will update once model finishes training]
* **F1 Score:** [Will update once model finishes training]
* **Mean Angle Error:** [Will update once model finishes training]

### Visual Output
*[Insert a screenshot here of your model drawing boxes on the tubes]*

---

## 💻 How to Run This Project

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
