FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces require a non-root user (UID 1000) with a
# writable HOME directory.
RUN useradd --create-home --shell /bin/bash --uid 1000 user

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app
RUN chown -R user:user /home/user

USER user

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

# HF Spaces injects $PORT (default 7860). server:api is the FastAPI app
# defined in server.py — exposes /reset, /step, /state and mounts the
# Vishwamitra Gradio UI at /ui.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:api --host 0.0.0.0 --port ${PORT:-7860}"]
