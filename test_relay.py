import serial
import time
import sys

# Replace 'COM3' with your actual Arduino COM port
# For Linux/Mac it might be '/dev/ttyUSB0' or '/dev/ttyACM0'
PORT = 'COM7' 
BAUD_RATE = 9600

try:
    print(f"Connecting to Arduino on {PORT}...")
    # Initialize serial connection
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=2)
    
    # Wait for Arduino to reset and initialize (takes ~2 seconds)
    time.sleep(2)
    
    # Read any startup messages
    if arduino.in_waiting:
        print("Arduino says:", arduino.read(arduino.in_waiting).decode('utf-8').strip())

    print("\nSending command to turn ON the relay...")
    arduino.write(b"DEV-LIGHT_ON\n")
    
    time.sleep(1)
    
    # Read response
    if arduino.in_waiting:
        print("Response:", arduino.read(arduino.in_waiting).decode('utf-8').strip())
    else:
        print("No response from Arduino. Please check your wiring, COM port, and make sure the Serial Monitor in Arduino IDE is CLOSED.")

    print("\nIf you want to turn it off, run the script again and change 'DEV-LIGHT_ON' to 'DEV-LIGHT_OFF'")

except serial.SerialException as e:
    print(f"Error: Could not open port {PORT}. Make sure the Arduino is plugged in and the port is correct.")
    print(f"Detailed error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("\nSerial connection closed.")
