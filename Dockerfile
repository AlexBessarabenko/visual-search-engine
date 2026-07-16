# Dockerfile — CPU-only образ с Python 3.11
FROM python:3.11-slim-bookworm

LABEL maintainer="AlexBessarabenko"
LABEL description="Visual Search Engine — Multimodal Image Search (CPU)"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ставим CPU-only PyTorch отдельно, чтобы не тянуть CUDA в образ
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu torchvision==0.21.0+cpu

# Копируем requirements и устанавливаем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY app/ ./app/
COPY notebooks/ ./notebooks/
COPY scripts/ ./scripts/
COPY plan.md .

# Создаём директории для данных
RUN mkdir -p data/images data/embeddings data/indexes

# PYTHONPATH=/app позволяет импортировать src и scripts из любой точки
ENV PYTHONPATH=/app

# По умолчанию запускаем bash (для интерактивной работы)
CMD ["/bin/bash"]
