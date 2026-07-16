"""
Веб-интерфейс на Streamlit для демонстрации поиска изображений.
Запуск: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from src import config
from src.dataset import load_metadata
from src.embeddings import load_jina_model, load_clip_model, encode_texts_jina, encode_texts_clip
from src.index import load_index, search_index
import numpy as np


@st.cache_resource
def load_cached_metadata():
    """Кэшируем метаданные."""
    return load_metadata()


@st.cache_resource
def load_cached_jina():
    """Кэшируем модель Jina."""
    return load_jina_model()


@st.cache_resource
def load_cached_clip():
    """Кэшируем модель CLIP."""
    return load_clip_model()


@st.cache_resource
def load_cached_index(index_path):
    """Кэшируем FAISS индекс."""
    return load_index(index_path)


def main():
    st.set_page_config(page_title="Visual Search Engine", layout="wide")
    
    st.title("🔍 Мультимодальный поиск изображений по тексту")
    st.markdown("Поиск изображений через **Jina v5-omni-nano** и **CLIP**")
    
    # Загрузка данных
    try:
        captions, image_paths = load_cached_metadata()
    except Exception as e:
        st.error(f"Ошибка загрузки метаданных: {e}")
        st.info("Сначала запустите пайплайн: `python -m src.dataset` и `python -m src.embeddings`")
        return
    
    # Sidebar — настройки
    st.sidebar.header("⚙️ Настройки")
    model_choice = st.sidebar.selectbox(
        "Модель",
        ["Jina v5-omni-nano", "CLIP (ViT-B/32)"],
    )
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=config.TOP_K_DEFAULT)
    
    # Выбор индекса и модели
    is_jina = model_choice == "Jina v5-omni-nano"
    index_path = config.FAISS_FLAT_JINA if is_jina else config.FAISS_FLAT_CLIP
    
    # Загрузка модели и индекса
    try:
        index = load_cached_index(index_path)
        if is_jina:
            model, processor = load_cached_jina()
        else:
            model, processor = load_cached_clip()
    except Exception as e:
        st.error(f"Ошибка загрузки модели/индекса: {e}")
        return
    
    # Ввод запроса
    st.header("📝 Введите запрос")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Текстовый запрос",
            value="a dog on the beach",
            placeholder="Например: a red car",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Искать", use_container_width=True)
    
    # Примеры запросов
    st.subheader("📌 Быстрые примеры")
    example_cols = st.columns(5)
    examples = config.TEST_QUERIES[:5]
    for i, ex in enumerate(examples):
        with example_cols[i]:
            if st.button(ex, key=f"ex_{i}"):
                query = ex
                search_btn = True
    
    # Поиск
    if search_btn and query:
        with st.spinner("Поиск..."):
            # Энкодинг запроса
            if is_jina:
                query_emb = encode_texts_jina([query], model, processor, is_query=True)
            else:
                query_emb = encode_texts_clip([query], model, processor)
            
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            
            # Поиск в FAISS
            distances, indices, search_time = search_index(index, query_emb, top_k=top_k)
        
        st.success(f"Поиск завершён за {search_time*1000:.1f} мс")
        
        # Отображение результатов
        st.header(f"📸 Результаты (Top-{top_k})")
        
        result_cols = st.columns(min(top_k, 5))
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            idx = int(idx)
            col_idx = (rank - 1) % 5
            with result_cols[col_idx]:
                img = Image.open(image_paths[idx]).convert("RGB")
                st.image(img, use_container_width=True)
                st.caption(f"**#{rank}** | Score: {score:.3f}")
                st.caption(f"_{captions[idx][:80]}..._")
    
    # Инфо
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 В индексе: {index.ntotal} изображений")
    st.sidebar.info(f"🖼️ Датасет: Conceptual Captions ({len(captions)})")


if __name__ == "__main__":
    main()
