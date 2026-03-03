import sys
import os

# Add 'src' to sys.path so that internal imports in src/ (like 'import inference_engine') work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.main_app import main
except ImportError as e:
    print(f"Error importing main_app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
