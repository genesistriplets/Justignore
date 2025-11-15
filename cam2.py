import cv2
import socket
import numpy as np
import struct
import sys
import time

# --- Network Configuration (Messenger/Server) ---
# Use '0.0.0.0' to listen on all available network interfaces
HOST = '0.0.0.0'
PORT = 9999

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

# --- Socket Server Setup ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server_socket.bind((HOST, PORT))
except socket.error as msg:
    print(f'Bind failed. Error Code : {str(msg[0])} Message {msg[1]}')
    sys.exit()

server_socket.listen(1)
print(f"Waiting for Receiver to connect on {socket.gethostbyname(socket.gethostname())}:{PORT}...")

conn, addr = None, None
try:
    conn, addr = server_socket.accept()
    print(f"Connection established with Receiver at {addr}. Starting stream...")
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

        # 1. Encode Frame to JPEG for efficient transmission
        # Quality: 80 (0-100) - determines compression/data size
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

        if not result:
            print("Could not encode frame.")
            continue

        data = encoded_frame.tobytes()

        # 2. Send Frame Size (4 bytes, unsigned long)
        msg_size = struct.pack("!L", len(data))
        conn.sendall(msg_size + data)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"Streamed raw frame size: {len(data) / 1024:.2f} KB | Capture FPS: {fps:.2f}", end='\r')

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
