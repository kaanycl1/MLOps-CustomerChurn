# Dockerfile
FROM python:3.11-slim

# Keeps logs unbuffered + avoids writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (catboost can need these in slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching)
# If you use requirements.api.txt, change the filename here.
COPY requirements.api.txt /app/requirements.api.txt
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy code + artifacts (model files must exist in repo)
COPY api.py predict.py /app/
COPY artifacts /app/artifacts

EXPOSE 8000

# Start API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
