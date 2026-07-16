"""
Конфигурация проекта visual-search-engine.
Все пути и гиперпараметры централизованы здесь.
"""

import os
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"

# Создаём директории при импорте
for d in (DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, INDEXES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Датасет ─────────────────────────────────────────────────────────────
DATASET_NAME = "google-research-datasets/conceptual_captions"
MAX_SAMPLES = 200                # CPU-friendly: ~200 изображений
IMAGE_TIMEOUT = 15               # Секунд на скачивание одного изображения
MAX_WORKERS = 4                  # Параллельных загрузчиков

# ─── Модели ──────────────────────────────────────────────────────────────
JINA_MODEL = "jinaai/jina-embeddings-v5-omni-nano"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Jina-специфичные настройки (оптимизация под CPU/16GB RAM)
JINA_MODALITY = "vision"         # Загружаем только vision + text towers
JINA_TRUNCATE_DIM = 256          # Снижаем размерность с 768 -> 256
JINA_BATCH_SIZE = 4              # Маленький батч для CPU
JINA_DTYPE = "float32"           # CPU не поддерживает bf16

# CLIP-специфичные настройки
CLIP_BATCH_SIZE = 16             # CLIP легче, можно больше батч

# ─── Эмбеддинги ──────────────────────────────────────────────────────────
EMBEDDING_DIM = JINA_TRUNCATE_DIM  # 256 для Jina, 512 для CLIP
CACHE_JINA_IMAGE = EMBEDDINGS_DIR / "jina_image_embeddings.npy"
CACHE_JINA_TEXT = EMBEDDINGS_DIR / "jina_text_embeddings.npy"
CACHE_CLIP_IMAGE = EMBEDDINGS_DIR / "clip_image_embeddings.npy"
CACHE_CLIP_TEXT = EMBEDDINGS_DIR / "clip_text_embeddings.npy"
CACHE_CAPTIONS = EMBEDDINGS_DIR / "captions.json"
CACHE_IMAGE_PATHS = EMBEDDINGS_DIR / "image_paths.json"

# ─── FAISS индекс ────────────────────────────────────────────────────────
FAISS_FLAT_JINA = INDEXES_DIR / "faiss_flat_jina.index"
FAISS_FLAT_CLIP = INDEXES_DIR / "faiss_flat_clip.index"
FAISS_HNSW_JINA = INDEXES_DIR / "faiss_hnsw_jina.index"
FAISS_HNSW_CLIP = INDEXES_DIR / "faiss_hnsw_clip.index"

# HNSW параметры
HNSW_M = 16       # Количество связей на узел
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# ─── Поиск ───────────────────────────────────────────────────────────────
TOP_K_DEFAULT = 5

# ─── Тестовые запросы ────────────────────────────────────────────────────
TEST_QUERIES = [
    "a dog on the beach",
    "a red car",
    "people playing football",
    "a cat sitting on a table",
    "a mountain landscape with snow",
    "a group of friends at a party",
    "a pizza on a wooden table",
    "a person riding a bicycle",
    "sunset over the ocean",
    "a baby sleeping in a crib",
]

# Русские варианты (для демо)
TEST_QUERIES_RU = [
    "собака на пляже",
    "красный автомобиль",
    "люди играют в футбол",
    "кот сидит на столе",
    "горный пейзаж со снегом",
    "группа друзей на вечеринке",
    "пицца на деревянном столе",
    "человек едет на велосипеде",
    "закат над океаном",
    "младенец спит в кроватке",
]
