# Use CUDA 12.9 base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV CHATTERBOX_DEVICE=cuda
RUN mkdir -p /app/audio_prompts
# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-full python3-pip git

# Upgrade pip and install requirements from file
COPY requirements.txt /app/requirements.txt
WORKDIR /app
#RUN pip install --upgrade pip
RUN pip install  --no-cache-dir --break-system-packages -r requirements.txt 
RUN pip install --pre --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Copy application code and entry script
COPY run.sh /app/run.sh
COPY main.py /app/main.py
RUN mkdir -p /app/completed

# Make shell script executable
RUN chmod +x /app/run.sh 
# Expose FastAPI on port 80
EXPOSE 80

# Set entrypoint
#CMD ["/app/run.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]