"""
Модуль поиска изображений по текстовым запросам.
Обёртка над FAISS индексом + ранжирование + визуализация.
"""

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import config
from .embeddings import encode_texts_jina, encode_texts_clip, load_jina_model, load_clip_model
from .index import load_index, search_index


# Вес ключевых слов в гибридном поиске (0 = чисто векторный, 1 = чисто ключевые слова)
HYBRID_KEYWORD_WEIGHT = 0.7
# Сколько кандидатов отбираем векторным поиском для последующего переранжирования
HYBRID_CANDIDATE_MULTIPLIER = 5


def compute_keyword_score(query: str, caption: str) -> float:
    """
    Доля слов запроса, найденных в подписи (substring matching, case-insensitive).
    """
    query_words = [w for w in query.lower().split() if w]
    if not query_words:
        return 0.0
    caption_lower = caption.lower()
    matched = sum(1 for w in query_words if w in caption_lower)
    return matched / len(query_words)


def normalize_scores(results: List[dict]) -> None:
    """Нормализует векторные score результатов в [0, 1] по min-max внутри выборки."""
    if not results:
        return
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s
    for r in results:
        r["vector_score_norm"] = (r["score"] - min_s) / rng if rng > 1e-9 else 1.0


def rerank_hybrid(results: List[dict], query: str, alpha: float = HYBRID_KEYWORD_WEIGHT) -> List[dict]:
    """
    Переранжирует кандидатов комбинацией векторного сходства и keyword overlap.
    """
    normalize_scores(results)
    for r in results:
        kw_score = compute_keyword_score(query, r["caption"])
        r["hybrid_score"] = alpha * kw_score + (1 - alpha) * r["vector_score_norm"]
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results


def search_images(
    query: str,
    index_path: Path,
    image_paths: List[Path],
    captions: List[str],
    model_name: str = "jina",
    top_k: int = config.TOP_K_DEFAULT,
    use_hybrid: bool = True,
) -> List[dict]:
    """
    Поиск изображений по текстовому запросу.
    
    Args:
        query: текстовый запрос
        index_path: путь к FAISS индексу
        image_paths: список путей к изображениям (для отображения)
        captions: список подписей (для отображения)
        model_name: "jina" или "clip"
        top_k: количество результатов
        use_hybrid: если True, применяет гибридное ранжирование (вектор + ключевые слова)
    
    Returns:
        Список результатов: [{"rank", "caption", "image_path", "score", "index"}, ...]
    """
    # Загружаем индекс
    index = load_index(index_path)
    
    # Создаём эмбеддинг запроса
    if model_name == "jina":
        model, processor = load_jina_model()
        query_emb = encode_texts_jina([query], model, processor, is_query=True)
    elif model_name == "clip":
        model, processor = load_clip_model()
        query_emb = encode_texts_clip([query], model, processor)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # L2-нормализация запроса (важно для IndexFlatIP)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # Векторный поиск: отбираем больше кандидатов для гибридного переранжирования
    candidate_k = top_k * HYBRID_CANDIDATE_MULTIPLIER if use_hybrid else top_k
    distances, indices, search_time = search_index(index, query_emb, top_k=candidate_k)
    
    # Формируем результаты
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        idx = int(idx)
        results.append({
            "rank": rank,
            "index": idx,
            "caption": captions[idx],
            "image_path": image_paths[idx],
            "score": float(score),  # cosine similarity
        })
    
    if use_hybrid:
        results = rerank_hybrid(results, query)
        # Оставляем только top_k и обновляем rank/score для отображения
        results = results[:top_k]
        for rank, r in enumerate(results, start=1):
            r["rank"] = rank
            r["score"] = r["hybrid_score"]
    
    return results, search_time


def display_results(
    query: str,
    results: List[dict],
    save_path: Path = None,
    figsize: Tuple[int, int] = (15, 4),
) -> None:
    """
    Визуализирует результаты поиска в виде сетки изображений.
    
    Args:
        query: исходный запрос (заголовок)
        results: список результатов из search_images
        save_path: если указан, сохраняет PNG
        figsize: размер фигуры matplotlib
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n / 5, figsize[1]))
    
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        img = Image.open(res["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"#{res['rank']}\nScore: {res['score']:.3f}\n{res['caption'][:50]}...")
        ax.axis("off")
    
    fig.suptitle(f'Query: "{query}"', fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Search] Результат сохранён: {save_path}")
    
    plt.show()


def batch_search(
    queries: List[str],
    index_path: Path,
    image_paths: List[Path],
    captions: List[str],
    model_name: str = "jina",
    top_k: int = config.TOP_K_DEFAULT,
    use_hybrid: bool = True,
) -> List[Tuple[List[dict], float]]:
    """
    Пакетный поиск по нескольким запросам.
    
    Returns:
        Список (results, search_time) для каждого запроса.
    """
    # Загружаем индекс и модель один раз
    index = load_index(index_path)
    
    if model_name == "jina":
        model, processor = load_jina_model()
        query_embs = encode_texts_jina(queries, model, processor, is_query=True)
    elif model_name == "clip":
        model, processor = load_clip_model()
        query_embs = encode_texts_clip(queries, model, processor)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # L2-нормализация
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    
    # Поиск
    candidate_k = top_k * HYBRID_CANDIDATE_MULTIPLIER if use_hybrid else top_k
    distances, indices, search_time = search_index(index, query_embs, top_k=candidate_k)
    
    # Формируем результаты
    all_results = []
    for q_idx, query in enumerate(queries):
        results = []
        for rank, (idx, score) in enumerate(zip(indices[q_idx], distances[q_idx]), start=1):
            idx = int(idx)
            results.append({
                "rank": rank,
                "index": idx,
                "caption": captions[idx],
                "image_path": image_paths[idx],
                "score": float(score),
            })
        if use_hybrid:
            results = rerank_hybrid(results, query)[:top_k]
            for rank, r in enumerate(results, start=1):
                r["rank"] = rank
                r["score"] = r["hybrid_score"]
        all_results.append((results, search_time / len(queries)))
    
    return all_results


if __name__ == "__main__":
    from .dataset import load_metadata
    
    captions, image_paths = load_metadata()
    
    # Тестовый поиск
    results, t = search_images(
        "a dog on the beach",
        config.FAISS_FLAT_JINA,
        image_paths,
        captions,
        model_name="jina",
        top_k=5,
    )
    
    print(f"Найдено за {t:.4f}s:")
    for r in results:
        print(f"  #{r['rank']}: score={r['score']:.3f} | {r['caption'][:60]}...")
