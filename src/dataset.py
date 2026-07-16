"""
Модуль загрузки и предобработки датасета Conceptual Captions.
Скачивает подвыборку изображений по URL и сохраняет локально.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from . import config


def load_conceptual_captions(num_samples: int = config.MAX_SAMPLES) -> Tuple[List[str], List[str]]:
    """
    Загружает подвыборку Conceptual Captions из HuggingFace.
    
    Returns:
        image_urls: список URL изображений
        captions: список текстовых подписей
    """
    print(f"[Dataset] Загрузка conceptual_captions (первые {num_samples} примеров)...")
    
    ds = load_dataset(
        config.DATASET_NAME,
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    image_urls = []
    captions = []
    
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        url = example.get("image_url") or example.get("url")
        caption = example.get("caption") or example.get("text")
        if url and caption:
            image_urls.append(url)
            captions.append(caption)
    
    print(f"[Dataset] Загружено {len(image_urls)} пар (URL, caption)")
    return image_urls, captions


def download_single_image(args: Tuple[int, str, Path]) -> Tuple[int, bool, Path]:
    """
    Скачивает одно изображение по URL.
    
    Args:
        args: (idx, url, save_dir)
    
    Returns:
        (idx, success, image_path)
    """
    idx, url, save_dir = args
    image_path = save_dir / f"img_{idx:05d}.jpg"
    
    if image_path.exists():
        # Проверяем валидность существующего файла
        try:
            with Image.open(image_path) as img:
                img.verify()
            return idx, True, image_path
        except Exception:
            pass  # Перезакачиваем
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=config.IMAGE_TIMEOUT, headers=headers)
        response.raise_for_status()
        
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        # Проверяем валидность
        with Image.open(image_path) as img:
            img.verify()
        
        return idx, True, image_path
    except Exception as e:
        # Удаляем битый файл
        if image_path.exists():
            os.remove(image_path)
        return idx, False, image_path


def download_images(image_urls: List[str], captions: List[str]) -> Tuple[List[str], List[str], List[Path]]:
    """
    Параллельное скачивание изображений с фильтрацией битых ссылок.
    
    Returns:
        valid_urls: рабочие URL
        valid_captions: соответствующие подписи
        valid_paths: локальные пути к изображениям
    """
    print(f"[Dataset] Скачивание {len(image_urls)} изображений ({config.MAX_WORKERS} потоков)...")
    
    args = [(i, url, config.IMAGES_DIR) for i, url in enumerate(image_urls)]
    
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        for result in tqdm(
            executor.map(download_single_image, args),
            total=len(args),
            desc="Downloading"
        ):
            results.append(result)
    
    # Фильтруем только успешные
    valid_indices = [idx for idx, success, _ in results if success]
    valid_urls = [image_urls[idx] for idx in valid_indices]
    valid_captions = [captions[idx] for idx in valid_indices]
    valid_paths = [config.IMAGES_DIR / f"img_{idx:05d}.jpg" for idx in valid_indices]
    
    print(f"[Dataset] Успешно скачано: {len(valid_paths)} / {len(image_urls)}")
    return valid_urls, valid_captions, valid_paths


def save_metadata(captions: List[str], image_paths: List[Path]) -> None:
    """Сохраняет метаданные в JSON для последующего использования."""
    with open(config.CACHE_CAPTIONS, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    path_strs = [str(p) for p in image_paths]
    with open(config.CACHE_IMAGE_PATHS, "w", encoding="utf-8") as f:
        json.dump(path_strs, f, indent=2)
    
    print(f"[Dataset] Метаданные сохранены: {config.CACHE_CAPTIONS}, {config.CACHE_IMAGE_PATHS}")


def load_metadata() -> Tuple[List[str], List[Path]]:
    """Загружает метаданные из JSON."""
    with open(config.CACHE_CAPTIONS, "r", encoding="utf-8") as f:
        captions = json.load(f)
    
    with open(config.CACHE_IMAGE_PATHS, "r", encoding="utf-8") as f:
        path_strs = json.load(f)
    
    image_paths = [Path(p) for p in path_strs]
    return captions, image_paths


def prepare_dataset(force_reload: bool = False) -> Tuple[List[str], List[Path]]:
    """
    Полный пайплайн подготовки датасета.
    
    Args:
        force_reload: Если True, игнорирует кэш и перезагружает всё
    
    Returns:
        captions, image_paths
    """
    if not force_reload and config.CACHE_CAPTIONS.exists() and config.CACHE_IMAGE_PATHS.exists():
        print("[Dataset] Используем кэшированные метаданные")
        return load_metadata()
    
    # Загружаем URL и подписи
    image_urls, captions = load_conceptual_captions()
    
    # Скачиваем изображения
    valid_urls, valid_captions, valid_paths = download_images(image_urls, captions)
    
    # Сохраняем метаданные
    save_metadata(valid_captions, valid_paths)
    
    return valid_captions, valid_paths


if __name__ == "__main__":
    prepare_dataset()
