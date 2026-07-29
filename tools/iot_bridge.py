import socketio
import serial
import json
import time
import os
import sys

# Change this to match your Arduino's COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
ARDUINO_PORT = 'COM7'
BAUD_RATE = 9600

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(CURRENT_DIR, 'sign_action_map.json')

# Initialize Socket.IO Client
sio = socketio.Client()

# Initialize Serial Connection
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=2)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2) # Wait for Arduino to reset after serial connection
except Exception as e:
    print(f"Warning: Could not connect to Arduino on {ARDUINO_PORT}. Error: {e}")
    print("Running in software-only mode (No hardware output).")
    arduino = None

# Load Action Map
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, 'r') as f:
        action_map = json.load(f)
    print(f"Loaded {len(action_map)} sign actions.")
else:
    print(f"Error: Could not find action map at {MAP_FILE}")
    sys.exit(1)


@sio.event
def connect():
    print("Connected to SignVision WebRTC Server!")

@sio.event
def disconnect():
    print("Disconnected from Server.")

@sio.on('prediction')
def on_prediction(data):
    cmd = data.get('text', '').lower()
    conf = float(data.get('conf', 0.0))
    print(f"Received Prediction: {cmd} (Conf: {conf:.2f})")
    
    if cmd in action_map:
        if cmd in ['bulb_on', 'bulb_off'] and conf < 0.65:
            print(f"  -> Ignoring {cmd} (Confidence {conf:.2f} < 0.65)")
            return
            
        mapping = action_map[cmd]
        device = mapping['device']
        action = mapping['action']
        
        # Format command for Arduino (e.g., "DEV-LIGHT_TOGGLE\n")
        serial_cmd = f"{device.upper()}_{action.upper()}\n"
        print(f"  -> Sending to Hardware: {serial_cmd.strip()}")
        
        if arduino and arduino.is_open:
            try:
                arduino.write(serial_cmd.encode('utf-8'))
            except serial.SerialException as e:
                print(f"  -> Serial Write Error (Arduino disconnected?): {e}")
    else:
        print(f"  -> No hardware mapping for sign: {cmd}")


if __name__ == '__main__':
    max_retries = 10
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to WebRTC Server (Attempt {attempt+1}/{max_retries})...")
            # Connect to the WebRTC Server (default is localhost:8080)
            sio.connect('http://localhost:8080', wait_timeout=15)
            sio.wait()
            break # Exit loop if wait() finishes cleanly
        except Exception as e:
            print(f"Failed to connect to Socket.IO server: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                if arduino and arduino.is_open:
                    arduino.close()

