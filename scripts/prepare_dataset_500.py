"""
Пересоздаёт датасет: запрашивает MAX_SAMPLES URL, скачивает,
сохраняет только первые TARGET_SAMPLES валидных изображений.
"""
from src import config
from src.dataset import prepare_dataset, save_metadata


def main():
    print(f"[Prepare] Целевой размер датасета: {config.TARGET_SAMPLES}")
    captions, image_paths = prepare_dataset(force_reload=True)

    if len(captions) > config.TARGET_SAMPLES:
        print(f"[Prepare] Обрезаем датасет с {len(captions)} до {config.TARGET_SAMPLES}")
        captions = captions[: config.TARGET_SAMPLES]
        image_paths = image_paths[: config.TARGET_SAMPLES]
        save_metadata(captions, image_paths)

    print(f"[Prepare] Итоговый датасет: {len(captions)} изображений")


if __name__ == "__main__":
    main()
