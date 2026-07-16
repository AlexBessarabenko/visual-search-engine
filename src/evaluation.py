"""
Модуль оценки качества поиска.
Метрики: Precision@k, Recall@k, mAP, среднее время поиска.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .embeddings import (
    encode_texts_jina,
    encode_texts_clip,
    load_jina_model,
    load_clip_model,
)
from .index import load_index, search_index
from .search import rerank_hybrid, HYBRID_CANDIDATE_MULTIPLIER


def compute_precision_at_k(
    retrieved_indices: np.ndarray,
    relevant_indices: set,
    k: int,
) -> float:
    """
    Precision@k = (релевантные в топ-k) / k
    
    Args:
        retrieved_indices: индексы найденных документов (отсортированы по релевантности)
        relevant_indices: множество "золотых" (релевантных) индексов
        k: позиция отсечения
    """
    top_k = retrieved_indices[:k]
    relevant_in_top_k = len(set(top_k) & relevant_indices)
    return relevant_in_top_k / k


def compute_recall_at_k(
    retrieved_indices: np.ndarray,
    relevant_indices: set,
    k: int,
) -> float:
    """
    Recall@k = (релевантные в топ-k) / (всего релевантных)
    """
    if len(relevant_indices) == 0:
        return 0.0
    top_k = retrieved_indices[:k]
    relevant_in_top_k = len(set(top_k) & relevant_indices)
    return relevant_in_top_k / len(relevant_indices)


def compute_ap(
    retrieved_indices: np.ndarray,
    relevant_indices: set,
) -> float:
    """
    Average Precision (AP) — усреднение Precision@k для каждой позиции,
    где найден релевантный документ.
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, idx in enumerate(retrieved_indices):
        if idx in relevant_indices:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    return sum_precisions / len(relevant_indices)


def get_relevant_indices(
    query: str,
    captions: List[str],
) -> set:
    """
    Определяет "релевантные" изображения для запроса.
    
    Стратегия: ищем подстроковое совпадение ключевых слов запроса в подписях.
    Это суррогатная "золотая" разметка (не идеальна, но объективна).
    """
    query_lower = query.lower()
    keywords = query_lower.split()
    
    relevant = set()
    for idx, caption in enumerate(captions):
        caption_lower = caption.lower()
        # Считаем релевантным, если хотя бы 1 ключевое слово найдено
        if any(kw in caption_lower for kw in keywords):
            relevant.add(idx)
    
    return relevant


def evaluate_model(
    queries: List[str],
    captions: List[str],
    image_paths: List[Path],
    index_path: Path,
    model_name: str = "jina",
    top_k_values: List[int] = [1, 3, 5, 10],
    use_hybrid: bool = True,
) -> pd.DataFrame:
    """
    Полная оценка модели на тестовых запросах.
    
    Args:
        use_hybrid: если True, применяет гибридное ранжирование (вектор + ключевые слова)
    
    Returns:
        DataFrame с метриками по каждому запросу и усреднённые.
    """
    # Загружаем индекс и модель
    index = load_index(index_path)
    
    if model_name == "jina":
        model, processor = load_jina_model()
        encode_fn = lambda texts: encode_texts_jina(texts, model, processor, is_query=True)
    elif model_name == "clip":
        model, processor = load_clip_model()
        encode_fn = lambda texts: encode_texts_clip(texts, model, processor)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Кодируем все запросы батчем
    query_embs = encode_fn(queries)
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    
    # Поиск
    max_k = max(top_k_values)
    candidate_k = max_k * HYBRID_CANDIDATE_MULTIPLIER if use_hybrid else max_k
    distances, indices, search_time = search_index(index, query_embs, top_k=candidate_k)
    
    # Считаем метрики
    rows = []
    for q_idx, query in enumerate(queries):
        retrieved_indices = indices[q_idx]
        
        if use_hybrid:
            # Переранжируем кандидатов с учётом ключевых слов
            candidates = [
                {
                    "index": int(idx),
                    "caption": captions[int(idx)],
                    "score": float(score),
                }
                for idx, score in zip(retrieved_indices, distances[q_idx])
            ]
            reranked = rerank_hybrid(candidates, query)
            retrieved_indices = np.array([r["index"] for r in reranked[:max_k]])
        
        relevant = get_relevant_indices(query, captions)
        
        row = {
            "query": query,
            "relevant_count": len(relevant),
        }
        
        for k in top_k_values:
            row[f"P@{k}"] = compute_precision_at_k(retrieved_indices, relevant, k)
            row[f"R@{k}"] = compute_recall_at_k(retrieved_indices, relevant, k)
        
        row["AP"] = compute_ap(retrieved_indices, relevant)
        row["search_time_ms"] = (search_time / len(queries)) * 1000
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Усреднённые метрики
    mean_row = {"query": "MEAN"}
    for k in top_k_values:
        mean_row[f"P@{k}"] = df[f"P@{k}"].mean()
        mean_row[f"R@{k}"] = df[f"R@{k}"].mean()
    mean_row["AP"] = df["AP"].mean()
    mean_row["mAP"] = mean_row["AP"]  # mAP = mean(AP) по запросам
    mean_row["search_time_ms"] = df["search_time_ms"].mean()
    
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    return df


def compare_models(
    queries: List[str],
    captions: List[str],
    image_paths: List[Path],
    top_k_values: List[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    """
    Сравнение Jina vs CLIP на одном наборе запросов.
    
    Returns:
        DataFrame со сравнением метрик.
    """
    print("[Eval] === Jina v5-omni-nano ===")
    df_jina = evaluate_model(
        queries, captions, image_paths,
        config.FAISS_FLAT_JINA, "jina", top_k_values
    )
    
    print("[Eval] === CLIP ===")
    df_clip = evaluate_model(
        queries, captions, image_paths,
        config.FAISS_FLAT_CLIP, "clip", top_k_values
    )
    
    # Извлекаем только средние строки
    jina_mean = df_jina[df_jina["query"] == "MEAN"].iloc[0].drop("query")
    clip_mean = df_clip[df_clip["query"] == "MEAN"].iloc[0].drop("query")
    
    comparison = pd.DataFrame({
        "Jina (v5-omni-nano)": jina_mean,
        "CLIP (ViT-B/32)": clip_mean,
    })
    
    comparison["Improvement"] = comparison["Jina (v5-omni-nano)"] - comparison["CLIP (ViT-B/32)"]
    
    print("\n[Eval] === Сравнение моделей ===")
    print(comparison.round(4))
    
    return comparison


def compare_index_types(
    queries: List[str],
    captions: List[str],
    image_paths: List[Path],
    model_name: str = "jina",
) -> pd.DataFrame:
    """
    Сравнение Flat vs HNSW индекса по скорости и качеству.
    """
    print(f"[Eval] === Сравнение индексов ({model_name}) ===")
    
    df_flat = evaluate_model(
        queries, captions, image_paths,
        config.FAISS_FLAT_JINA if model_name == "jina" else config.FAISS_FLAT_CLIP,
        model_name, top_k_values=[5, 10]
    )
    
    df_hnsw = evaluate_model(
        queries, captions, image_paths,
        config.FAISS_HNSW_JINA if model_name == "jina" else config.FAISS_HNSW_CLIP,
        model_name, top_k_values=[5, 10]
    )
    
    flat_mean = df_flat[df_flat["query"] == "MEAN"].iloc[0]
    hnsw_mean = df_hnsw[df_hnsw["query"] == "MEAN"].iloc[0]
    
    comparison = pd.DataFrame({
        "Flat": flat_mean,
        "HNSW": hnsw_mean,
    }).drop("query")
    
    comparison["Speedup"] = comparison["Flat"] / comparison["HNSW"]  # Для времени
    
    print(comparison.round(4))
    return comparison


if __name__ == "__main__":
    from .dataset import load_metadata
    
    captions, image_paths = load_metadata()
    
    # Сравнение моделей
    comparison = compare_models(config.TEST_QUERIES, captions, image_paths)
    print(comparison)
