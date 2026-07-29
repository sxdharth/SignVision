#!/usr/bin/env python3
"""
Backward-compatibility wrapper for launching the SignVision Smart Home suite.
This delegates directly to the unified CLI launcher: `python signvision.py --mode web`
"""

import sys
import os
from signvision import run_web_mode, print_banner

if __name__ == "__main__":
    print_banner()
    run_web_mode(port=8080)
