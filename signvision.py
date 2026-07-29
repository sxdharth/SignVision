#!/usr/bin/env python3
"""
SignVision Unified CLI Launcher
===============================
An enterprise-grade CLI entry point for launching the SignVision platform modes:
  - web     : Launches the asynchronous WebRTC video calling server & Arduino IoT Bridge.
  - desktop : Launches the local standalone OpenCV sign detection window.
  - iot     : Launches standalone Arduino Serial Bridge for Hardware-in-the-Loop automation.

Usage:
    python signvision.py --mode web
    python signvision.py --mode desktop
    python signvision.py --mode iot
    python signvision.py --help
"""

import argparse
import os
import sys
import time
import socket
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_banner():
    print(r"""
  ____  _             __     __ _       _                 
 / ___|(_)  __ _  _ __ \ \   / /(_) ___ (_)  ___  _ __    
 \___ \| | / _` || '_ \ \ \ / / | |/ __|| | / _ \| '_ \   
  ___) | || (_| || | | | \ V /  | |\__ \| || (_) || | | | 
 |____/|_| \__, ||_| |_|  \_/   |_||___/|_| \___/ |_| |_| 
           |___/                                          
  AI-Powered Accessible Communication & Smart Home IoT Suite
    """)


def check_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def force_kill_port(port: int):
    print(f"[!] Port {port} is locked by a lingering process! Releasing port...")
    if os.name == 'nt':
        os.system(f"FOR /F \"tokens=5\" %T IN ('netstat -ano ^| findstr :{port}') DO taskkill /F /PID %T >nul 2>&1")
    time.sleep(1.5)


def run_web_mode(port: int = 8080):
    print("=========================================================")
    print(" [~] Starting SignVision Full Web Platform & IoT Suite [~] ")
    print("=========================================================")

    if check_port_in_use(port):
        force_kill_port(port)

    print(f"\n[1/2] Launching Asynchronous WebRTC Signaling Server on Port {port}...")
    server_process = subprocess.Popen(
        [sys.executable, os.path.join("web", "webrtc", "server.py")],
        cwd=ROOT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    # Essential delay to allow WebSocket port binding and TensorFlow model weight loading
    time.sleep(12)

    print("\n[2/2] Launching Hardware-in-the-Loop (HITL) Arduino IoT Bridge...")
    bridge_process = subprocess.Popen(
        [sys.executable, os.path.join("tools", "iot_bridge.py")],
        cwd=ROOT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    try:
        print("\n[SUCCESS] SIGNVISION PLATFORM IS FULLY ONLINE")
        print(f"  >> Core Portal   : http://localhost:{port}/")
        print(f"  >> Video Call    : http://localhost:{port}/call")
        print(f"  >> Smart Home    : http://localhost:{port}/smart_home.html")
        print("\n  >> Press [Ctrl+C] to safely shut down all platform services.\n")

        server_process.wait()
        bridge_process.wait()

    except KeyboardInterrupt:
        print("\n[STOP] Shutdown signal received! Terminating services gracefully...")
        server_process.terminate()
        bridge_process.terminate()
        server_process.wait()
        bridge_process.wait()
        print("Shutdown complete. Goodbye!")


def run_desktop_mode():
    print("=========================================================")
    print("      [~] Starting Standalone Desktop Detection [~]      ")
    print("=========================================================")
    src_dir = os.path.join(ROOT_DIR, 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    try:
        from src.main_app import main
        main()
    except ImportError as e:
        print(f"[ERROR] Failed to import desktop application: {e}")
        sys.exit(1)


def run_iot_mode():
    print("=========================================================")
    print("     [~] Starting Standalone Arduino Serial Bridge [~]   ")
    print("=========================================================")
    bridge_path = os.path.join(ROOT_DIR, 'tools', 'iot_bridge.py')
    try:
        subprocess.run([sys.executable, bridge_path], cwd=ROOT_DIR, check=True)
    except KeyboardInterrupt:
        print("\n[STOP] Arduino IoT bridge shut down.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignVision: AI-Powered Multimodal Communication & Smart Home IoT Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["web", "desktop", "iot"],
        default="web",
        help="Select the operating mode to launch:\n"
             "  web     - Launch the WebRTC video calling server + IoT Arduino Bridge (Default)\n"
             "  desktop - Launch the local standalone OpenCV sign language detector\n"
             "  iot     - Launch standalone Arduino serial relay bridge daemon"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="HTTP/WebSocket port for web mode (Default: 8080)"
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    if args.mode == "web":
        run_web_mode(port=args.port)
    elif args.mode == "desktop":
        run_desktop_mode()
    elif args.mode == "iot":
        run_iot_mode()


if __name__ == "__main__":
    main()
