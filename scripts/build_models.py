"""Script to build TF-IDF vectorizer/matrix and SBERT embeddings from dataset.

Usage:
    python scripts/build_models.py --data data/clean_movies.csv --out-models models/

This will:
 - create `models/tfidf_vectorizer.joblib` and `models/tfidf_matrix.npz`
 - create `models/sbert_embeddings.npy` (via SentenceTransformer)
"""
import os
import argparse
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

# Import preprocessing helper from src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import load_and_process_data


def build_tfidf(df, text_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    texts = df[text_col].fillna('').astype(str).tolist()
    X = vect.fit_transform(texts)
    vect_path = os.path.join(out_dir, 'tfidf_vectorizer.joblib')
    mat_path = os.path.join(out_dir, 'tfidf_matrix.npz')
    joblib.dump(vect, vect_path)
    sp.save_npz(mat_path, X)
    print(f"Saved TF-IDF vectorizer -> {vect_path}")
    print(f"Saved TF-IDF matrix -> {mat_path}")


def build_sbert(df, text_col, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    texts = df[text_col].fillna('').astype(str).tolist()
    embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=True)
    emb_arr = np.array(embeddings)
    emb_path = os.path.join(out_dir, 'sbert_embeddings.npy')
    np.save(emb_path, emb_arr)
    print(f"Saved SBERT embeddings -> {emb_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/clean_movies.csv')
    parser.add_argument('--out-models', default='models')
    parser.add_argument('--sbert-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--tfidf-text-col', default='similarity_text')
    parser.add_argument('--sbert-text-col', default='full_text')
    args = parser.parse_args()

    print('Loading and processing data...')
    df = load_and_process_data(args.data)
    if df.empty:
        print('No data loaded; aborting')
        return

    print('Building TF-IDF...')
    build_tfidf(df, args.tfidf_text_col, args.out_models)

    print('Building SBERT embeddings...')
    build_sbert(df, args.sbert_text_col, args.sbert_model, args.out_models)

    print('All done.')


if __name__ == '__main__':
    main()
