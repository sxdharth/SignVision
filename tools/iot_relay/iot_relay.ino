// SignVision IoT Relay Controller
// Active-LOW Relay Module Support (change `LOW` to `HIGH` if yours is Active-HIGH)

const int RELAY_PIN = 7; // Connect Relay IN to Pin 7
String cmd;
bool relayState = false; // Track current state for toggling

void setup() {
    Serial.begin(9600);
    
    // Initialize Relay Pin
    pinMode(RELAY_PIN, OUTPUT);
    
    // Default to OFF for Active-LOW relay (HIGH means OFF)
    // If your relay turns ON when the Arduino powers up, change this to LOW.
    digitalWrite(RELAY_PIN, HIGH);
    
    Serial.println("SignVision Node Started.");
}

void loop() {
    if (Serial.available()) {
        cmd = Serial.readStringUntil('\n');
        cmd.trim(); // Remove '\r' or spaces
        
        if (cmd == "DEV-LIGHT_TOGGLE") {
            // Toggle state
            relayState = !relayState;
            
            // For active-LOW relays, LOW = ON, HIGH = OFF
            if (relayState) {
                digitalWrite(RELAY_PIN, LOW); // Turn ON
                Serial.println("LIGHT_IS_ON");
            } else {
                digitalWrite(RELAY_PIN, HIGH); // Turn OFF
                Serial.println("LIGHT_IS_OFF");
            }
        }
        else if (cmd == "DEV-LIGHT_ON") {
            relayState = true;
            digitalWrite(RELAY_PIN, LOW);
            Serial.println("LIGHT_IS_ON");
        }
        else if (cmd == "DEV-LIGHT_OFF") {
            relayState = false;
            digitalWrite(RELAY_PIN, HIGH);
            Serial.println("LIGHT_IS_OFF");
        }
    }
}
