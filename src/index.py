"""
Модуль работы с FAISS векторной БД.
Поддерживает два типа индексов:
  - Flat: точный поиск (brute-force), медленнее на больших данных
  - HNSW: приближённый поиск (ANN), быстрый, чуть менее точный
"""

import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from . import config


def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Строит FAISS Flat-индекс (exact search).
    
    Args:
        embeddings: (N, D) матрица эмбеддингов, L2-нормализованная
    
    Returns:
        FAISS IndexFlatIP (Inner Product = cosine similarity для нормализованных векторов)
    """
    dim = embeddings.shape[1]
    
    # IndexFlatIP = Inner Product. Для нормализованных векторов IP = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"[Index] Flat index построен: {index.ntotal} векторов, dim={dim}")
    return index


def build_hnsw_index(
    embeddings: np.ndarray,
    m: int = config.HNSW_M,
    ef_construction: int = config.HNSW_EF_CONSTRUCTION,
) -> faiss.Index:
    """
    Строит FAISS HNSW-индекс (approximate nearest neighbor).
    
    Args:
        embeddings: (N, D) матрица эмбеддингов
        m: количество связей на узел
        ef_construction: качество построения графа
    
    Returns:
        FAISS IndexHNSWFlat
    """
    dim = embeddings.shape[1]
    
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    
    print(f"[Index] HNSW index построен: {index.ntotal} векторов, "
          f"m={m}, efConstruction={ef_construction}")
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """Сохраняет FAISS индекс на диск."""
    faiss.write_index(index, str(path))
    print(f"[Index] Индекс сохранён: {path}")


def load_index(path: Path) -> faiss.Index:
    """Загружает FAISS индекс с диска."""
    index = faiss.read_index(str(path))
    print(f"[Index] Индекс загружен: {path} ({index.ntotal} векторов)")
    return index


def search_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = config.TOP_K_DEFAULT,
    ef_search: int = config.HNSW_EF_SEARCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поиск ближайших соседей в FAISS индексе.
    
    Args:
        index: FAISS индекс
        query_embeddings: (Q, D) матрица запросов
        top_k: количество результатов
        ef_search: параметр для HNSW (игнорируется для Flat)
    
    Returns:
        distances: (Q, top_k) — косинусные сходства (для IP index)
        indices: (Q, top_k) — индексы найденных векторов
    """
    # Для HNSW устанавливаем efSearch
    if hasattr(index, 'hnsw'):
        index.hnsw.efSearch = ef_search
    
    start = time.time()
    distances, indices = index.search(query_embeddings, top_k)
    elapsed = time.time() - start
    
    print(f"[Index] Поиск завершён: {query_embeddings.shape[0]} запросов, "
          f"top_k={top_k}, time={elapsed:.4f}s")
    
    return distances, indices, elapsed


def build_or_load_index(
    embeddings: np.ndarray,
    index_path: Path,
    index_type: str = "flat",
    force_rebuild: bool = False,
) -> faiss.Index:
    """
    Универсальная функция: загружает существующий индекс или строит новый.
    
    Args:
        embeddings: эмбеддинги для индексации
        index_path: путь сохранения/загрузки
        index_type: "flat" или "hnsw"
        force_rebuild: принудительно перестроить
    
    Returns:
        FAISS Index
    """
    if not force_rebuild and index_path.exists():
        return load_index(index_path)
    
    if index_type == "flat":
        index = build_flat_index(embeddings)
    elif index_type == "hnsw":
        index = build_hnsw_index(embeddings)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    save_index(index, index_path)
    return index


if __name__ == "__main__":
    import numpy as np
    
    # Тест
    emb = np.random.randn(100, 256).astype("float32")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2-нормализация
    
    idx = build_flat_index(emb)
    print(f"Test flat: {idx.ntotal} vectors")
    
    idx_hnsw = build_hnsw_index(emb)
    print(f"Test HNSW: {idx_hnsw.ntotal} vectors")
