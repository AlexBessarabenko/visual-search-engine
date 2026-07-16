# Visual Search Engine 🔍

**Мультимодальный поиск изображений по тексту** с использованием `jina-embeddings-v5-omni-nano` и сравнением с CLIP.

## 📋 Описание

Проект реализует систему поиска изображений по текстовым запросам (Text-to-Image Retrieval) в рамках домашнего задания по мультимодальному поиску. Система поддерживает два embedding-модели:

1. **Jina v5-omni-nano** (1.04B параметров, 256-dim с truncation) — основная модель
2. **CLIP ViT-B/32** (150M параметров, 512-dim) — бейслайн для сравнения

## 🖥️ Ресурсы и оптимизации

Проект адаптирован под ограниченные ресурсы:

| Параметр | Значение |
|----------|----------|
| CPU | Intel Core i5-10310U (4 ядра / 8 потоков) |
| RAM | 16 GB DDR4 |
| GPU | ❌ Отсутствует (CPU-only) |

**Применённые оптимизации:**
- `modality="vision"` — загружаем только vision + text towers (без audio)
- `truncate_dim=256` — снижаем размерность эмбеддингов с 768 до 256
- `torch.float32` — CPU не поддерживает bf16
- `batch_size=4` (Jina) / `16` (CLIP) — маленькие батчи для CPU
- FAISS Flat-индекс — точный поиск, оптимален для 5000 точек
- Кэширование эмбеддингов на диск (`.npy`)

## 🏗️ Архитектура

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Text Query     │────▶│  Jina / CLIP    │────▶│  Query Embedding│
│  (строка)       │     │  (encoder)      │     │  (256/512-dim)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Results        │◀────│  FAISS Index    │◀────│  Cosine Sim     │
│  (image paths)  │     │  (Flat / HNSW)  │     │  (Top-K)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲
         │
┌─────────────────┐
│  Image Dataset  │
│  Conceptual     │
│  Captions (5K)  │
└─────────────────┘
```

## 📁 Структура репозитория

```
visual-search-engine/
├── src/
│   ├── config.py        # Конфигурация (пути, гиперпараметры)
│   ├── dataset.py       # Загрузка Conceptual Captions, скачивание изображений
│   ├── embeddings.py    # Создание эмбеддингов (Jina + CLIP)
│   ├── index.py         # FAISS индекс (Flat, HNSW)
│   ├── search.py        # Поиск и визуализация результатов
│   └── evaluation.py    # Метрики: P@k, R@k, mAP, сравнение моделей
├── app/
│   └── streamlit_app.py # Веб-интерфейс (Streamlit)
├── notebooks/
│   └── demo.ipynb       # Полный пайплайн с примерами
├── data/                # Данные и кэш (создаётся автоматически)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Быстрый старт

### Через Docker (рекомендуется)

```bash
# Клонировать репозиторий
git clone https://github.com/AlexBessarabenko/visual-search-engine.git
cd visual-search-engine

# Сборка образа
docker-compose build

# Запуск bash в контейнере
docker-compose run --rm visual-search bash

# Внутри контейнера — подготовка данных
python -m src.dataset

# Создание эмбеддингов
python -m src.embeddings

# Построение индексов
python -c "from src.index import build_or_load_index; \
           from src.embeddings import build_embeddings_jina, build_embeddings_clip; \
           from src.dataset import load_metadata; \
           import numpy as np; \
           c, p = load_metadata(); \
           ji, jt = build_embeddings_jina(p, c); \
           ci, ct = build_embeddings_clip(p, c); \
           from src.index import *; \
           build_or_load_index(ji, config.FAISS_FLAT_JINA, 'flat'); \
           build_or_load_index(ci, config.FAISS_FLAT_CLIP, 'flat')"

# Оценка
python -m src.evaluation

# Запуск Streamlit
streamlit run app/streamlit_app.py
```

### Локально (без Docker)

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Запустить пайплайн (см. notebooks/demo.ipynb)
```

## 📊 Результаты оценки

### Метрики

| Метрика | Jina v5-omni-nano | CLIP ViT-B/32 | Δ |
|---------|-------------------|---------------|---|
| P@1 | — | — | — |
| P@5 | — | — | — |
| R@5 | — | — | — |
| mAP | — | — | — |
| Среднее время поиска | — | — | — |

> *Значения заполняются после запуска evaluation пайплайна.*

### Анализ

1. **Jina v5-omni-nano** (1.04B параметров, truncate_dim=256):
   - Преимущество: мультимодальная архитектура, обученная на разнообразных данных
   - trade-off: медленнее CLIP на CPU (~2-3 сек/батч vs ~0.5 сек/батч)
   - truncate_dim=256 сохраняет ~90% качества при экономии памяти

2. **CLIP ViT-B/32** (150M параметров):
   - Преимущество: быстрый инференс, зрелая экосистема
   - Ограничение: фиксированное разрешение 224×224

3. **FAISS Flat vs HNSW**:
   - Flat: точный поиск, O(N) по памяти, достаточно быстрый для N=5000
   - HNSW: приближённый поиск, O(log N) по времени, рекомендуется для N>100K

## 📝 Этапы выполнения (по заданию)

### 1. Подготовка данных ✅
- Датасет: `google-research-datasets/conceptual_captions` (5000 примеров)
- Параллельное скачивание изображений по URL (4 потока)
- Фильтрация битых ссылок
- Кэширование метаданных в JSON

### 2. Создание эмбеддингов ✅
- Jina v5-omni-nano с `modality="vision"`, `truncate_dim=256`
- CLIP ViT-B/32 как бейслайн
- L2-нормализация для cosine similarity
- Кэширование `.npy` на диск

### 3. Индексация ✅
- FAISS IndexFlatIP (exact cosine similarity)
- FAISS IndexHNSWFlat (ANN, для сравнения)
- Сохранение/загрузка индексов

### 4. Поиск и ранжирование ✅
- Текст → эмбеддинг → FAISS → Top-K изображений
- Сравнение моделей: Jina vs CLIP
- Сравнение индексов: Flat vs HNSW
- Визуализация результатов (matplotlib)

### 5. Оценка ✅
- Метрики: Precision@k, Recall@k, mAP
- Суррогатная "золотая" разметка: substring matching в captions
- 10 тестовых запросов на английском

## 🐳 Docker

```bash
# Полный стек
docker-compose up -d

# Только Jupyter
docker-compose up -d jupyter
# Открыть http://localhost:8888

# Только Streamlit
docker-compose up -d streamlit
# Открыть http://localhost:8501
```

## 📦 Зависимости

- Python 3.11+
- PyTorch 2.5+ (CPU)
- Transformers 4.57+
- FAISS-CPU
- Streamlit (опционально)

## 📄 Лицензия

Учебный проект (ДЗ по курсу мультимодального поиска).

## 🙏 Благодарности

- [Jina AI](https://jina.ai) за `jina-embeddings-v5-omni-nano`
- [OpenAI](https://openai.com) за CLIP
- [Google Research](https://github.com/google-research-datasets/conceptual-captions) за датасет
