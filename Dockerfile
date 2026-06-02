# ============================================================================
# Cat or Dog – Inference API
# Multi-stage–ready, slim Python image for production serving
# ============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- Install dependencies (layer-cached) ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Copy application code ------------------------------------------
COPY model/ ./model/
COPY app/   ./app/
COPY weights/ ./weights/

# ---------- Runtime ---------------------------------------------------------
# Hugging Face Spaces require port 7860 and a non-root user (id 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

ENV PORT=7860
EXPOSE $PORT

# Run the ASGI server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
