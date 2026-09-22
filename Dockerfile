# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - libgl1, libglib2.0-0: required for opencv-python
# - espeak, alsa-utils: required for pyttsx3 (TTS) on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    espeak \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We also upgrade pip for faster and more reliable installs
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the web app runs on
EXPOSE 8080

# Set the default command to launch the unified CLI in web mode
CMD ["python", "signvision.py", "--mode", "web", "--port", "8080"]
