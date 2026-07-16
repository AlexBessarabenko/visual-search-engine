"""
Модуль создания мультимодальных эмбеддингов.
Поддерживает две модели:
  1. Jina v5-omni-nano (vision modality, truncate_dim=256) — основная
  2. CLIP (openai/clip-vit-base-patch32) — бейслайн для сравнения
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from . import config


def load_jina_model() -> Tuple[AutoModel, AutoProcessor]:
    """
    Загружает Jina v5-omni-nano с оптимизациями для CPU.
    
    Оптимизации:
    - modality="vision" — загружаем только vision + text towers
    - truncate_dim=256 — снижаем размерность эмбеддингов
    - float32 — CPU не поддерживает bf16
    """
    print(f"[Embeddings] Загрузка {config.JINA_MODEL} (modality={config.JINA_MODALITY})...")
    
    device = torch.device("cpu")
    
    model = AutoModel.from_pretrained(
        config.JINA_MODEL,
        trust_remote_code=True,
        modality=config.JINA_MODALITY,
        dtype=torch.float32,
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(
        config.JINA_MODEL,
        trust_remote_code=True,
    )
    
    # Для retrieval задачи используем encode_document для изображений
    # и encode_query для текстовых запросов
    
    print(f"[Embeddings] Модель загружена. Device: {device}")
    return model, processor


def load_clip_model() -> Tuple[CLIPModel, CLIPProcessor]:
    """Загружает CLIP для сравнения."""
    print(f"[Embeddings] Загрузка {config.CLIP_MODEL}...")
    
    device = torch.device("cpu")
    
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    
    print(f"[Embeddings] CLIP загружен. Device: {device}")
    return model, processor


def encode_images_jina(
    image_paths: List[Path],
    model: AutoModel,
    processor: AutoProcessor,
    batch_size: int = config.JINA_BATCH_SIZE,
) -> np.ndarray:
    """
    Создаёт эмбеддинги изображений через Jina v5-omni-nano.
    Используем encode_document для индексации изображений.
    """
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Jina image encoding"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # Jina через sentence-transformers API: encode_document для документов
        # На уровне transformers: proc(images=..., text="Document: <image>")
        inputs = processor(
            images=images,
            text=["Document: <image>"] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        with torch.no_grad():
            # truncate_dim=256 для экономии памяти
            batch_emb = model.embed(**inputs, truncate_dim=config.JINA_TRUNCATE_DIM)
        
        embeddings.append(batch_emb.cpu().numpy())
    
    return np.vstack(embeddings).astype("float32")


def encode_texts_jina(
    texts: List[str],
    model: AutoModel,
    processor: AutoProcessor,
    batch_size: int = config.JINA_BATCH_SIZE,
    is_query: bool = True,
) -> np.ndarray:
    """
    Создаёт эмбеддинги текстов через Jina.
    Для запросов: "Query: текст"
    Для документов: "Document: текст"
    """
    prefix = "Query: " if is_query else "Document: "
    prefixed_texts = [prefix + t for t in texts]
    
    embeddings = []
    
    for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Jina text encoding"):
        batch_texts = prefixed_texts[i:i + batch_size]
        
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            batch_emb = model.embed(**inputs, truncate_dim=config.JINA_TRUNCATE_DIM)
        
        embeddings.append(batch_emb.cpu().numpy())
    
    return np.vstack(embeddings).astype("float32")


def encode_images_clip(
    image_paths: List[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int = config.CLIP_BATCH_SIZE,
) -> np.ndarray:
    """Создаёт эмбеддинги изображений через CLIP."""
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP image encoding"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).pooler_output
            # L2-нормализация (CLIP эмбеддинги уже нормализованы, но на всякий случай)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings.append(image_features.cpu().numpy())
    
    return np.vstack(embeddings).astype("float32")


def encode_texts_clip(
    texts: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int = config.CLIP_BATCH_SIZE,
) -> np.ndarray:
    """Создаёт эмбеддинги текстов через CLIP."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text encoding"):
        batch_texts = texts[i:i + batch_size]
        
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(embeddings).astype("float32")


def build_embeddings_jina(
    image_paths: List[Path],
    captions: List[str],
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создаёт/загружает кэшированные эмбеддинги Jina.
    
    Returns:
        (image_embeddings, text_embeddings)
    """
    if (not force_rebuild and 
        config.CACHE_JINA_IMAGE.exists() and 
        config.CACHE_JINA_TEXT.exists()):
        print("[Embeddings] Загрузка кэша Jina...")
        img_emb = np.load(config.CACHE_JINA_IMAGE)
        txt_emb = np.load(config.CACHE_JINA_TEXT)
        return img_emb, txt_emb
    
    model, processor = load_jina_model()
    
    print("[Embeddings] Создание image embeddings (Jina)...")
    img_emb = encode_images_jina(image_paths, model, processor)
    
    print("[Embeddings] Создание text embeddings (Jina)...")
    txt_emb = encode_texts_jina(captions, model, processor, is_query=False)
    
    # Сохраняем кэш
    np.save(config.CACHE_JINA_IMAGE, img_emb)
    np.save(config.CACHE_JINA_TEXT, txt_emb)
    
    print(f"[Embeddings] Jina embeddings сохранены: shape={img_emb.shape}")
    return img_emb, txt_emb


def build_embeddings_clip(
    image_paths: List[Path],
    captions: List[str],
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создаёт/загружает кэшированные эмбеддинги CLIP.
    
    Returns:
        (image_embeddings, text_embeddings)
    """
    if (not force_rebuild and 
        config.CACHE_CLIP_IMAGE.exists() and 
        config.CACHE_CLIP_TEXT.exists()):
        print("[Embeddings] Загрузка кэша CLIP...")
        img_emb = np.load(config.CACHE_CLIP_IMAGE)
        txt_emb = np.load(config.CACHE_CLIP_TEXT)
        return img_emb, txt_emb
    
    model, processor = load_clip_model()
    
    print("[Embeddings] Создание image embeddings (CLIP)...")
    img_emb = encode_images_clip(image_paths, model, processor)
    
    print("[Embeddings] Создание text embeddings (CLIP)...")
    txt_emb = encode_texts_clip(captions, model, processor)
    
    # Сохраняем кэш
    np.save(config.CACHE_CLIP_IMAGE, img_emb)
    np.save(config.CACHE_CLIP_TEXT, txt_emb)
    
    print(f"[Embeddings] CLIP embeddings сохранены: shape={img_emb.shape}")
    return img_emb, txt_emb


if __name__ == "__main__":
    from .dataset import load_metadata
    
    captions, image_paths = load_metadata()
    
    # Jina
    jina_img, jina_txt = build_embeddings_jina(image_paths, captions)
    print(f"Jina: image={jina_img.shape}, text={jina_txt.shape}")
    
    # CLIP
    clip_img, clip_txt = build_embeddings_clip(image_paths, captions)
    print(f"CLIP: image={clip_img.shape}, text={clip_txt.shape}")
