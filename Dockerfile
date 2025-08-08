FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lung_cancer_detector/ ./lung_cancer_detector/

ENV PYTHONPATH=/app/lung_cancer_detector

CMD ["python", "lung_cancer_detector/app/app.py"]