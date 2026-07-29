// SignVision IoT Relay Controller
// ACTIVE-LOW RELAY MODULE SUPPORT
// This code is specifically for relays that turn ON when the signal is LOW.

const int RELAY_PIN = 7;     // Connect Light Relay IN to Pin 7
const int FAN_PIN = 8;       // Connect Fan Relay IN to Pin 8
const int BOARD_LED = 13;    // Built-in Arduino LED

String cmd;
bool relayState = false; // False = OFF, True = ON

void setup() {
    Serial.begin(9600);
    
    // For Active-LOW relays, we MUST write HIGH to the pins BEFORE setting them 
    // to OUTPUT. If we don't, they default to LOW and the lights turn on instantly.
    digitalWrite(RELAY_PIN, HIGH);
    digitalWrite(FAN_PIN, HIGH);
    
    // Now it's safe to make them outputs.
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(FAN_PIN, OUTPUT);
    
    pinMode(BOARD_LED, OUTPUT);
    digitalWrite(BOARD_LED, LOW); // Board LED is standard Active-HIGH
    
    Serial.println("SignVision Node Started.");
}

void loop() {
    if (Serial.available()) {
        cmd = Serial.readStringUntil('\n');
        cmd.trim(); 
        
        // Active-LOW Logic: HIGH = OFF, LOW = ON
        if (cmd == "DEV-LIGHT_TOGGLE") {
            relayState = !relayState;
            if (relayState) {
                digitalWrite(RELAY_PIN, LOW); // Turn ON
                digitalWrite(BOARD_LED, HIGH);
                Serial.println("LIGHT_IS_ON");
            } else {
                digitalWrite(RELAY_PIN, HIGH); // Turn OFF
                digitalWrite(BOARD_LED, LOW);
                Serial.println("LIGHT_IS_OFF");
            }
        }
        else if (cmd == "DEV-LIGHT_ON") {
            relayState = true;
            digitalWrite(RELAY_PIN, LOW); // Turn ON
            digitalWrite(BOARD_LED, HIGH);
            Serial.println("LIGHT_IS_ON");
        }
        else if (cmd == "DEV-LIGHT_OFF") {
            relayState = false;
            digitalWrite(RELAY_PIN, HIGH); // Turn OFF
            digitalWrite(BOARD_LED, LOW);
            Serial.println("LIGHT_IS_OFF");
        }
        else if (cmd == "DEV-FAN_ON") {
            digitalWrite(FAN_PIN, LOW); // Turn ON
            digitalWrite(BOARD_LED, HIGH);
            Serial.println("FAN_IS_ON");
        }
        else if (cmd == "DEV-FAN_OFF") {
            digitalWrite(FAN_PIN, HIGH); // Turn OFF
            digitalWrite(BOARD_LED, LOW);
            Serial.println("FAN_IS_OFF");
        }
    }
}
