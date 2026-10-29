FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY data/re24.json ./data/re24.json
# Editable installation keeps config.py's project root at /app, beside config.yaml.
RUN pip install --no-cache-dir --editable . \
    && mkdir -p models cache runs \
    && chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_TEMP_DIR=/app/runs/uploads \
    MPLCONFIGDIR=/app/cache/matplotlib \
    YOLO_CONFIG_DIR=/app/cache/yolo
# Download official weights into the image; no user videos or API keys are copied.
RUN python scripts/download_models.py
EXPOSE 7860
CMD ["python", "scripts/app.py", "--host", "0.0.0.0", "--port", "7860", "--max-file-size", "200mb"]
