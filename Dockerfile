# Use an official Python image with Debian (which supports FFmpeg)
FROM python:3.11-slim

# Install FFmpeg and required system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Start Flask when the container starts
CMD ["python", "app.py"]
