import cv2
import socket
import numpy as np
from ultralytics import YOLO
import struct
import sys
import time

# --- Network Configuration (Messenger/Server) ---
# Use '0.0.0.0' to listen on all available network interfaces
HOST = '0.0.0.0'
PORT = 9999

# --- YOLO Model Setup ---
print("Initializing YOLO Model...")
try:
    # Load YOLOv8 small model (downloads automatically on first run)
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit()

# COCO bottle class index
BOTTLE_CLASS_ID = 39

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

# --- Socket Server Setup ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Allows the socket to be reused immediately after close
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server_socket.bind((HOST, PORT))
except socket.error as msg:
    print(f'Bind failed. Error Code : {str(msg[0])} Message {msg[1]}')
    sys.exit()

server_socket.listen(1)
print(f"Waiting for connection on {socket.gethostbyname(socket.gethostname())}:{PORT}...")
print("Please run the Receiver script now.")

conn, addr = None, None
try:
    conn, addr = server_socket.accept()
    print(f"Connection established with Receiver at {addr}")
except Exception as e:
    print(f"Error during socket accept: {e}")
    server_socket.close()
    sys.exit()


# Main streaming loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, stopping.")
            break

        start_time = time.time()

        # 1. YOLO Detection (Processing on Messenger/Camera Computer)
        results = model.predict(
            frame,
            conf=0.5,
            classes=[BOTTLE_CLASS_ID],
            verbose=False,
            # Use smaller image size for faster processing on low-power device
            imgsz=320
        )

        # 2. Draw Boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Bottle {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3. Encode Frame to JPEG for efficient transmission
        # Quality: 80 (0-100, higher is better quality, larger file size)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

        if not result:
            print("Could not encode frame.")
            continue

        data = encoded_frame.tobytes()

        # 4. Send Frame Size (4 bytes, unsigned long)
        # This tells the receiver how much data to expect for the frame
        msg_size = struct.pack("!L", len(data))
        conn.sendall(msg_size + data)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"Streamed frame size: {len(data) / 1024:.2f} KB | FPS: {fps:.2f}", end='\r')

        # Press 'e' to exit the messenger
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

except ConnectionResetError:
    print("\nReceiver disconnected unexpectedly.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Cleanup
    print("\nClosing connections and camera...")
    if conn:
        conn.close()
    if server_socket:
        server_socket.close()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
