# CPU-only image for Nextflow -profile docker (or Singularity built from this image).
# Build from repository root: docker build -t aiupred:cpu .
FROM python:3.11-slim

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "numpy>=1.21" "scipy>=1.7"

WORKDIR /opt/aiupred
COPY . /opt/aiupred
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT []
