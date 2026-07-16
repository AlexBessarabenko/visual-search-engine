import numpy as np
from src.index import build_or_load_index
from src import config


def main():
    print('Building indexes...')
    jina_img = np.load(config.CACHE_JINA_IMAGE)
    clip_img = np.load(config.CACHE_CLIP_IMAGE)

    build_or_load_index(jina_img, config.FAISS_FLAT_JINA, 'flat')
    build_or_load_index(jina_img, config.FAISS_HNSW_JINA, 'hnsw')
    build_or_load_index(clip_img, config.FAISS_FLAT_CLIP, 'flat')
    build_or_load_index(clip_img, config.FAISS_HNSW_CLIP, 'hnsw')
    print('Done')


if __name__ == "__main__":
    main()
