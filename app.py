import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.recommender_tfidf import load_tfidf_models
from src.recommender_sbert import load_sbert_models

# Config paths
DATA_PATH = 'data/clean_movies.csv'
TFIDF_MODEL_PATH = 'models/tfidf_vectorizer.joblib'
TFIDF_MATRIX_PATH = 'models/tfidf_matrix.npz'
SBERT_MODEL_PATH = 'all-MiniLM-L6-v2'  # use name or folder
SBERT_EMB_PATH = 'models/sbert_embeddings.npy'

st.set_page_config(layout='wide', page_title='Semantic Movie Recommender')

st.title('🎬 Semantic Movie Recommender')
st.write('Demo app that loads TF-IDF and SBERT models (cached).')


@st.cache_resource
def load_models():
    tfidf_rec = load_tfidf_models(TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH, DATA_PATH)
    sbert_rec = load_sbert_models(SBERT_MODEL_PATH, SBERT_EMB_PATH, DATA_PATH)
    return tfidf_rec, sbert_rec


with st.spinner('Loading models (this may take a while on first run)...'):
    tfidf_rec, sbert_rec = load_models()

if tfidf_rec is None or sbert_rec is None:
    st.error('Could not load one or more models. Run `scripts/build_models.py` first to generate artifacts or check model paths.')
    st.stop()


tab1, tab2 = st.tabs(['Recommendation', 'Semantic Search'])

with tab1:
    st.header('Movie-to-Movie Recommendation')
    movie_list = tfidf_rec.df['title'].tolist()
    chosen = st.selectbox('Choose a movie', movie_list)
    k = st.slider('Top K', 1, 20, 10)
    if st.button('Get Recommendations'):
        with st.spinner('Computing...'):
            tfidf_results = tfidf_rec.get_similar_movies(chosen, top_k=k)
            sbert_results = sbert_rec.get_similar_movies(chosen, top_k=k)
        st.subheader('TF-IDF (Baseline)')
        st.table(pd.DataFrame(tfidf_results))
        st.subheader('SBERT (Semantic)')
        st.table(pd.DataFrame(sbert_results))

with tab2:
    st.header('Semantic Search')
    q = st.text_input('Enter query', '')
    k = st.slider('Top K', 1, 20, 10, key='search_k')
    if st.button('Search'):
        with st.spinner('Searching...'):
            tfidf_search = tfidf_rec.search_movies(q, top_k=k)
            sbert_search = sbert_rec.search_movies(q, top_k=k)
        st.subheader('TF-IDF (Keyword)')
        st.table(pd.DataFrame(tfidf_search))
        st.subheader('SBERT (Semantic)')
        st.table(pd.DataFrame(sbert_search))

st.sidebar.markdown('---')
st.sidebar.info('If models are missing, run: `python scripts/build_models.py`')
