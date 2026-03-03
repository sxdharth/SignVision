import serial
import time
import sys

ARDUINO_PORT = 'COM7'
BAUD_RATE = 9600

def test_arduino():
    print(f"Connecting to Arduino on {ARDUINO_PORT}...")
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=2)
        # Wait for the Arduino to reset and initialize after serial connection opens
        time.sleep(2)
        
        # Read the startup message if any
        if arduino.in_waiting > 0:
            msg = arduino.readline().decode('utf-8').strip()
            print(f"[Arduino]: {msg}")
            
    except serial.SerialException as e:
        print(f"Error: Could not open {ARDUINO_PORT}.")
        print(f"Details: {e}")
        print("Make sure the Arduino IDE Serial Monitor is closed and the board is plugged in.")
        sys.exit(1)

    print("\n--- Arduino Relay Test Menu ---")
    print("1: Turn Bulb ON  (Sends 'DEV-LIGHT_ON')")
    print("2: Turn Bulb OFF (Sends 'DEV-LIGHT_OFF')")
    print("3: Toggle Bulb   (Sends 'DEV-LIGHT_TOGGLE')")
    print("q: Quit")
    
    try:
        while True:
            choice = input("\nEnter choice: ").strip().lower()
            
            if choice == '1':
                cmd = "DEV-LIGHT_ON\n"
                print(f"Sending: {cmd.strip()}")
                arduino.write(cmd.encode('utf-8'))
            elif choice == '2':
                cmd = "DEV-LIGHT_OFF\n"
                print(f"Sending: {cmd.strip()}")
                arduino.write(cmd.encode('utf-8'))
            elif choice == '3':
                cmd = "DEV-LIGHT_TOGGLE\n"
                print(f"Sending: {cmd.strip()}")
                arduino.write(cmd.encode('utf-8'))
            elif choice == 'q':
                print("Exiting test script.")
                break
            else:
                print("Invalid choice.")
                continue
                
            # Wait briefly for Arduino to process and respond
            time.sleep(0.5)
            
            # Read all available responses from Arduino
            while arduino.in_waiting > 0:
                resp = arduino.readline().decode('utf-8').strip()
                print(f"  -> [Arduino response]: {resp}")
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        if arduino.is_open:
            arduino.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    test_arduino()
