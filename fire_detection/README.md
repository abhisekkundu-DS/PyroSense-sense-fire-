# FIRE Detection

This project, **Fire Detection**, is a computer vision-based YOLO application designed to detect fire in static images and video footage. It leverages OpenCV for processing and a model YOLO-v8  smoking and non fire classification  model to identify individuals fire in real-time or recorded environments.

## Folder Structure

- `test_image/` – Contains input test images  
- `test_video/` – Contains input test videos  
- `output_image/` – Stores output images with detection results  
- `output_video/` – Stores output videos with detection results  
- `fire_detection.py` – Main Python script to run detection  
- `fire_detection/` – Model directory (must include weights/configs)

## How to Run

1. Clone or download the repository
2. Ensure all dependencies (OpenCV, torch, etc.) are installed
3. Place your image or video in the respective input folder
4. Run the following command for:

### Image:
```bash
fire_detection.py test.jpg --model best_fire_detect.pt
```
### video :
```bash
fire_detection.py fire_test1.MP4 --model best_fire_detect.pt

## 🔥 Fire Detection Demo

![Fire Detection Demo](https://github.com/abhisekkundu-DS/PyroSense-sense-fire-/blob/main/fire_detection/main/redme_gif.gif?raw=true)

<p align="center">
  <img src="https://github.com/abhisekkundu-DS/PyroSense-sense-fire-/blob/main/fire_detection/main/redme_gif.gif?raw=true" alt="Fire Detection Demo">
</p>
