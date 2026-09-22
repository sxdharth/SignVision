import requests
import base64

graph = """graph TD
    subgraph Client ["<b>Web Browser Client (HTML5 / WebRTC / JS)</b>"]
        Cam[("Webcam Stream<br>(640x480 @ 15-20 FPS)")]
        RTC[("WebRTC P2P Video")]
        Captions["Live Captions & TTS Engine"]
    end

    subgraph Server ["<b>Async Python Signaling Server (AIOHTTP + Socket.IO)</b>"]
        Socket["WebSocket Router"]
        MP["1. MediaPipe Holistic Extractor<br>(166 3D Coordinates)"]
        Variance["2. Anti-Stationary Filter<br>(Variance &sigma;&sup2; < 0.005)"]
        RNN["3. Temporal Classifier<br>(Stacked Conv1D + GRU Network)"]
        NMT["4. Multilingual Translation<br>(Google NMT Engine)"]
    end

    subgraph Hardware ["<b>Smart Home IoT Bridge (pyserial)</b>"]
        Bridge["Python Serial Bridge Daemon"]
        MCU["Arduino Uno / Nano Relay Controller"]
        Relays["Physical Appliances (Lights / Devices)"]
    end

    Cam -->|JPEG Frames| Socket
    Socket --> MP
    MP --> Variance
    Variance -->|Active 30-Frame Sequence| RNN
    RNN -->|Sign Class & Confidence| NMT
    NMT -->|Translated Text| Socket
    Socket -->|Caption Display| Captions
    Socket <==>|WebRTC Signaling| RTC
    Socket -->|IoT Command Event| Bridge
    Bridge -->|RS-232 / USB Serial| MCU
    MCU -->|Relay Switching| Relays
"""

graphbytes = graph.encode("utf-8")
base64_bytes = base64.urlsafe_b64encode(graphbytes)
base64_string = base64_bytes.decode("utf-8")

# using mermaid.ink to render
url = "https://mermaid.ink/img/" + base64_string

print("Fetching image from:", url)
response = requests.get(url)
if response.status_code == 200:
    with open("architecture.png", "wb") as f:
        f.write(response.content)
    print("Successfully saved architecture.png")
else:
    print(f"Failed to fetch image. Status: {response.status_code}")
    print(response.text)
