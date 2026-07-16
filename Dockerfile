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

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY app/ ./app/
COPY notebooks/ ./notebooks/
COPY plan.md .

# Создаём директории для данных
RUN mkdir -p data/images data/embeddings data/indexes

# По умолчанию запускаем bash (для интерактивной работы)
CMD ["/bin/bash"]
