from src.dataset import load_metadata
from src.embeddings import build_embeddings_jina, build_embeddings_clip


def main():
    captions, image_paths = load_metadata()
    print(f"[Embeddings] Датасет: {len(image_paths)} изображений")

    print("[Embeddings] Jina...")
    jina_img, jina_txt = build_embeddings_jina(image_paths, captions, force_rebuild=True)
    print(f"Jina: {jina_img.shape}, {jina_txt.shape}")

    print("[Embeddings] CLIP...")
    clip_img, clip_txt = build_embeddings_clip(image_paths, captions, force_rebuild=True)
    print(f"CLIP: {clip_img.shape}, {clip_txt.shape}")


if __name__ == "__main__":
    main()
