import json
import re
from collections import Counter
from src.dataset import load_metadata
from src.search import search_images
from src import config


def main():
    with open(config.CACHE_CAPTIONS, encoding='utf-8') as f:
        captions = json.load(f)

    print('=== Примеры подписей (первые 15) ===')
    for i in range(15):
        print(f'{i}: {captions[i][:100]}')

    print('\n=== Частые осмысленные слова ===')
    stop = {'this', 'that', 'with', 'from', 'background', 'illustration', 'vector',
            'isolated', 'image', 'style', 'white', 'black', 'blue', 'red', 'green',
            'yellow', 'person', 'people', 'beautiful', 'pattern'}
    words = [w.lower() for c in captions for w in re.findall(r'[a-zA-Z]+', c)
             if len(w) > 3 and w.lower() not in stop]
    for w, n in Counter(words).most_common(30):
        print(f'{w}: {n}')

    queries = ["person on white background", "vector illustration", "film character",
               "summer beach", "beautiful woman", "seamless pattern",
               "dog", "car", "cat", "mountain"]
    print('\n=== Поиск: топ-3 подписи ===')
    image_paths, _ = load_metadata()
    for q in queries:
        print(f'\nQuery: {q}')
        kws = q.lower().split()
        matches = sum(1 for c in captions if any(kw in c.lower() for kw in kws))
        print(f'  keyword matches: {matches}')
        for model in ['jina', 'clip']:
            try:
                idx_path = config.FAISS_FLAT_JINA if model == 'jina' else config.FAISS_FLAT_CLIP
                results, _ = search_images(q, idx_path, image_paths, captions, model_name=model, top_k=3)
                print(f'  {model}: ' + ' | '.join([r['caption'][:50] for r in results]))
            except Exception as e:
                print(f'  {model} error: {e}')


if __name__ == "__main__":
    main()
