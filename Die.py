import cv2
import torch
from ultralytics import YOLO

# SETTINGS FOR SPEED
MODEL_NAME = "yolov10n.pt"  # Use YOLOv10 nano
SKIP_FRAMES = 2             # Inference every N frames (increase for more speed)
CONF_THRESH = 0.45          # Slightly lower conf for faster "good enough" detection
IMG_SIZE = 320              # Resize input: 320 is MUCH faster than 640
BOTTLE_CLASS_ID = 39        # COCO index for bottle

def run_detection():
    # 1. Load Model
    # Note: The first run will download yolov10n.pt automatically
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # 2. Check Device (Enable Half Precision if GPU is available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = device == 'cuda'
    print(f"Running on {device.upper()} with Half-Precision: {use_half}")

    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    # Lower webcam resolution at hardware level to reduce transfer overhead
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    last_boxes = [] # Store boxes to draw during skipped frames

    print("Starting detection... Press 'q' to exit.")
    
    try:
        while True:
            success, frame = cap.read()
            if not success: 
                break

            # LOGIC: Run AI only every 'SKIP_FRAMES' to save massive resources
            if frame_count % SKIP_FRAMES == 0:
                # Run inference
                results = model.predict(
                    source=frame,
                    classes=[BOTTLE_CLASS_ID], # Only detect bottles
                    conf=CONF_THRESH,
                    imgsz=IMG_SIZE,            # 320px is very fast
                    half=use_half,             # FP16 optimization
                    device=device,
                    max_det=10,                # Limit max detections to save post-process
                    verbose=False,             # Stop logging to console (saves I/O time)
                    stream=False               # Standard inference
                )
                
                # Update cached boxes
                last_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            # Draw the (possibly cached) boxes
            # We draw manually instead of using result.plot() which is slower
            for box in last_boxes:
                x1, y1, x2, y2 = box
                # Draw minimal box - Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Minimal label to avoid heavy font rendering
                cv2.putText(frame, "Bottle", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Fast YOLOv10n", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()

