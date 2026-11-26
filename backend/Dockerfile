# Build từ thư mục gốc repo:  docker build -f backend/Dockerfile -t asl-api .
FROM python:3.11-slim
WORKDIR /srv

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Chỉ lấy phần cần cho inference từ project của Triết
COPY models/Triet/src/__init__.py models/Triet/src/__init__.py
COPY models/Triet/src/preprocessing models/Triet/src/preprocessing
COPY models/Triet/src/feature_extraction models/Triet/src/feature_extraction
COPY models/Triet/models/production/exp3_mlp_opt.onnx models/Triet/models/production/
COPY models/Triet/models/production/exp3_mlp_opt_metadata.json models/Triet/models/production/
COPY backend/app backend/app

WORKDIR /srv/backend
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
