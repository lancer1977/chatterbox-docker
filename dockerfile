# Use CUDA 12.9 base image
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-full git

# Upgrade pip and install requirements from file
COPY requirements.txt /app/requirements.txt
WORKDIR /app
#RUN pip install --upgrade pip
RUN pip install --break-system-packages -r requirements.txt

# Copy application code and entry script
COPY run.sh /app/run.sh
COPY main.py /app/main.py

# Make shell script executable
RUN chmod +x /app/run.sh

# Expose FastAPI on port 80
EXPOSE 80

# Set entrypoint
CMD ["/app/run.sh"]
