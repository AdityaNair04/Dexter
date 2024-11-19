import socket
import pickle
import struct
import cv2
from ultralytics import YOLO
from pynput import keyboard

# Load YOLO model on the laptop
model = YOLO('yolov8n.pt')  # Load a lightweight model for inference

# Create a server socket to receive the video stream from Kria
stream_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
stream_socket.bind(('0.0.0.0', 9999))
stream_socket.listen(5)
print("Waiting for video stream...")

client_socket, addr = stream_socket.accept()
print(f"Connected to Kria at {addr}")

data = b""
payload_size = struct.calcsize("L")

# Set up socket to send movement commands back to the Kria
results_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
results_socket.connect(('192.168.2.1', 9998))  # Replace with Kria's IP

# YOLO class ID for bottle (use class 39 for the COCO dataset)
BOTTLE_CLASS_ID = 39

# To store the command based on keypress
command = "c"  # Default command is "center" (no movement)

# Function to send the movement command
def send_command():
    if command != "c":  # Only send commands when not centered
        results_socket.sendall(command.encode('utf-8'))

# Callback function for keyboard press
def on_press(key):
    global command
    try:
        if key.char == 'w':  # Move forward
            command = 'w'
        elif key.char == 'a':  # Move left
            command = 'a'
        elif key.char == 's':  # Move backward
            command = 's'
        elif key.char == 'd':  # Move right
            command = 'd'
        elif key.char == 'x':  # Custom command for 'x'
            command = 'x'
        elif key.char == 'm':  # Custom command for 'm'
            command = 'm'
    except AttributeError:
        if key == keyboard.Key.up:  # Up arrow key
            command = 'f'
        elif key == keyboard.Key.down:  # Down arrow key
            command = 'b'
        elif key == keyboard.Key.left:  # Left arrow key
            command = 'l'
        elif key == keyboard.Key.right:  # Right arrow key
            command = 'r'
    send_command()

# Callback function for keyboard release
def on_release(key):
    global command
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right] or \
       key.char in ['w', 'a', 's', 'd', 'x', 'm']:
        command = "c"  # Reset to "center" when key is released
        send_command()

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    # Retrieve message size
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    
    # Retrieve frame data
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Deserialize the frame using pickle
    frame = pickle.loads(frame_data)
    frame_height, frame_width = frame.shape[:2]
    
    # Run YOLO inference on the frame
    results = model(frame)
    
    # Filter results to only detect bottles
    bottle_found = False
    
    for result in results:
        filtered_boxes = []
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == BOTTLE_CLASS_ID:  # Check if detected object is a bottle
                filtered_boxes.append(box)
                bottle_found = True
                break  # Stop after processing the first bottle
    
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)  # Convert to integer values
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
    
        if bottle_found:
            break  # Stop further processing once a bottle is detected
    
    # (Optional) Display the inference results on the laptop for visualization
    cv2.imshow('YOLO Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close sockets and cleanup
cv2.destroyAllWindows()
client_socket.close()
results_socket.close()
listener.stop()
