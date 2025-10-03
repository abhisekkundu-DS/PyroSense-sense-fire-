from ultralytics import YOLO
import torch

import cv2
from collections import Counter

# Fire detection class names and colors
class_names = ['Fire']
class_colors = {'Fire': (0, 0, 255)}  # Red for fire

CONFIDENCE_THRESHOLD = 0.25  # keep detections above this

def detect_in_images(image_path, output_path, model, show=False, show_conf=False):
    image = cv2.imread(image_path)
    fire_detected = False
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Run model (ultralytics YOLO accepts a numpy image)
    results = model(image, imgsz=640)

    class_count = Counter()

    # iterate over results and boxes
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
            cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
            class_name = class_names[cls]

            # filter by confidence
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # label text (optionally show confidence)
            label = f"{class_name} {conf:.2f}" if show_conf else class_name
            color = class_colors.get(class_name, (0, 0, 255))  # default red

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            class_count[class_name] += 1

            # If fire detected
            if class_name == "Fire" and conf > CONFIDENCE_THRESHOLD:
                fire_detected = True

    # Write summary text at top
    summary_text = "FIRE: YES" if fire_detected else "FIRE: NO"
    summary_color = (0, 0, 255) if fire_detected else (0, 255, 0)
    cv2.putText(image, summary_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, summary_color, 3)

    # Optional: print summary counts
    if class_count:
        print("Detections summary:", dict(class_count))
    else:
        print("No detections above confidence threshold")

    # Save result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image to: {output_path}")

    # Show image if requested
    if show:
        cv2.imshow("Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_in_video(video_path, output_path, model, show=False, show_conf=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    CONFIDENCE_THRESHOLD = 0.25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fire = False  #  Initialize for each frame
        class_count = Counter()

        results = model.predict(frame, imgsz=640, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls = int(box.cls[0])
                class_name = class_names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{class_name} {conf:.2f}" if show_conf else class_name
                color = class_colors.get(class_name, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                class_count[class_name] += 1

                #  Case-insensitive match
                if class_name.lower() == "fire" and conf > CONFIDENCE_THRESHOLD:
                    fire = True

        summary_text = "FIRE: YES" if fire else "FIRE: NO"
        summary_color = (0, 0, 255) if fire else (0, 255, 0)
        cv2.putText(frame, summary_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, summary_color, 3)

        out.write(frame)

        if show:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved at: {output_path}")

import os
import argparse
from ultralytics import YOLO

# Import your custom detection functions
# Make sure detect_in_images and detect_in_video are defined in this file or imported
# from your detection_module import detect_in_images, detect_in_video

def main():
    parser = argparse.ArgumentParser(description="Fire detection using YOLOv8")
    parser.add_argument('input', type=str, help="Path to image or video input")
    parser.add_argument('--output', type=str, help="Path to save output (optional)")
    parser.add_argument('--model', default="best_fire_detect.pt", help="Path to trained YOLOv8 fire detection model")
    parser.add_argument('--show', action='store_true', help="Display output preview")
    parser.add_argument('--show_conf', action='store_true', help="Display confidence scores")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(" Input file not found:", args.input)
        return

    # Load YOLOv8 fire detection model
    model = YOLO(args.model)

    # Check file type
    _, ext = os.path.splitext(args.input.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        output = args.output or "fire_detected_image.jpg"
        detect_in_images(args.input, output, model, show=args.show, show_conf=args.show_conf)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        output = args.output or "fire_detected_video.mp4"
        detect_in_video(args.input, output, model, show=args.show, show_conf=args.show_conf)
    else:
        print(f"⚠ Unsupported file format: {ext}")


if __name__ == "__main__":
    main()
