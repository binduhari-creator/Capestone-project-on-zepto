FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and policy documents
COPY docs ./docs
COPY app ./app

# Expose service port
EXPOSE 7860

# Run FastAPI service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
