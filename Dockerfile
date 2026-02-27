# Use official Python lightweight image
FROM python:3.9-slim

# Expose Streamlit default port
EXPOSE 8501

# Install system utilities and OpenCV required dependencies
# (libgl1-mesa-glx or libglib2.0-0 are often needed even for headless opencv)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement list and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy everything else from the repo
COPY . .

# Run the Streamlit app
# --server.address=0.0.0.0 is crucial for Docker/Render deployments!
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
