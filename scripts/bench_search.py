import numpy as np
import faiss
from src.index import load_index
from src import config
import time

np.random.seed(42)
for name, path in [('Jina', config.FAISS_FLAT_JINA), ('CLIP', config.FAISS_FLAT_CLIP)]:
    idx = load_index(path)
    dim = idx.d
    queries = np.random.randn(10, dim).astype('float32')
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    idx.search(queries, 5)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        idx.search(queries, 5)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) / 10 * 1000
    print(f'{name}: per query {avg_ms:.3f} ms (total 10 queries {np.mean(times)*1000:.3f} ms)')
