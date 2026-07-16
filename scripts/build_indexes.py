import numpy as np
from src.index import build_or_load_index
from src import config


def main():
    print('Building indexes...')
    jina_img = np.load(config.CACHE_JINA_IMAGE)
    clip_img = np.load(config.CACHE_CLIP_IMAGE)

    build_or_load_index(jina_img, config.FAISS_FLAT_JINA, 'flat', force_rebuild=True)
    build_or_load_index(jina_img, config.FAISS_HNSW_JINA, 'hnsw', force_rebuild=True)
    build_or_load_index(clip_img, config.FAISS_FLAT_CLIP, 'flat', force_rebuild=True)
    build_or_load_index(clip_img, config.FAISS_HNSW_CLIP, 'hnsw', force_rebuild=True)
    print('Done')


if __name__ == "__main__":
    main()
