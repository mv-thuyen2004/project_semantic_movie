import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# FIX LỖI PATH: Thêm project root vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import lớp recommender và hàm load
from src.recommender_tfidf import load_tfidf_models
# ❌ LOẠI BỎ IMPORT SBERT


# 🔧 CONFIG CHUNG
DATA_PATH = 'data/clean_movies.csv'
TFIDF_MODEL_PATH = 'models/tfidf_vectorizer.pkl'
TFIDF_MATRIX_PATH = 'models/tfidf_matrix.npy' # Hoặc .joblib
# ❌ LOẠI BỎ CONFIG SBERT


# =======================================================
# 1. TẢI VÀ CACHING ASSETS (CHỈ TF-IDF)
# =======================================================

@st.cache_resource
def load_models_for_app():
    """Tải chỉ mô hình TF-IDF."""
    
    # 1. TF-IDF Recommender
    tfidf_recommender = load_tfidf_models(TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH, DATA_PATH)
    
    # ✅ KHÔNG TẢI SBERT
    sbert_recommender = None 
    
    return tfidf_recommender, sbert_recommender

tfidf_rec, sbert_rec = load_models_for_app()


# =======================================================
# 2. HÀM HỖ TRỢ HIỂN THỊ KẾT QUẢ (GIỮ NGUYÊN)
# =======================================================

def display_rich_results(results, model_name):
    """Hiển thị kết quả gợi ý với Poster và Score (Đã fix lỗi Year 'N/A')."""
    st.subheader(f"🏆 {model_name}")
    
    if not results:
        st.info(f"Không tìm thấy kết quả nào từ mô hình {model_name}.")
        return

    # Lặp qua từng phim trong kết quả
    for i, movie in enumerate(results):
        
        # Lấy giá trị năm an toàn
        raw_year = movie.get('year', 'N/A')
        
        # 1. LOGIC SỬA CHỮA: Kiểm tra và ép kiểu an toàn
        if pd.notna(raw_year) and raw_year != 'N/A':
             # Ép kiểu thành int chỉ khi giá trị là số hợp lệ
            display_year = int(raw_year) 
        else:
            display_year = 'N/A'
            
        # Chia cột cho Poster và Thông tin
        col_img, col_info = st.columns([1, 4]) 
        
        with col_img:
            # st.image hiển thị ảnh từ URL
            if movie.get('poster'):
                st.image(movie['poster'], width=100) 

                
            
        with col_info:
            # SỬ DỤNG BIẾN ĐÃ XỬ LÝ
            st.markdown(f"**{i+1}. {movie['title']}** ({display_year})")
            st.caption(f"Thể loại: {movie['genre']}")
            st.write(f"Điểm: **{movie['similarity_score']:.4f}**")
        
        st.markdown("---") # Phân cách giữa các phim

# =======================================================
# 3. LAYOUT CHÍNH CỦA ỨNG DỤNG
# =======================================================

st.set_page_config(layout="wide", page_title="Demo TF-IDF")
st.title("🎬 Demo Hệ thống Gợi ý Phim TF-IDF (Baseline)")
st.markdown("---")


# Kiểm tra trạng thái tải
if tfidf_rec is None:
    st.error("❌ LỖI: Không thể tải mô hình TF-IDF. Vui lòng kiểm tra file models.")
    st.stop()


# Sử dụng cấu trúc Tab cũ nhưng chỉ hiện thị TF-IDF trong cột 1
tab1, tab2 = st.tabs(["🎯 Gợi ý Phim Tương tự (Movie-to-Movie)", "🔍 Tìm kiếm Từ khóa (Keyword Search)"])


# --- TAB 1: GỢI Ý PHIM TƯƠNG TỰ (MOVIE-to-MOVIE) ---

with tab1:
    st.header("Gợi ý Phim Tương tự (TF-IDF)")
    
    movie_list = tfidf_rec.df['title'].tolist()
    
    selected_movie = st.selectbox(
        "Chọn một bộ phim để tìm kiếm sự tương đồng:",
        options=movie_list,
        index=0
    )
    
    top_k = st.slider("Số lượng kết quả gợi ý (Top K):", min_value=5, max_value=20, value=10)

    if selected_movie:
        # Lấy kết quả từ TF-IDF (Content-Based)
        tfidf_results = tfidf_rec.get_similar_movies(selected_movie, top_k=top_k)
        
        # CHỈ HIỂN THỊ CỘT TF-IDF
        col1, col2 = st.columns(2)
        
        with col1:
            display_rich_results(tfidf_results, "TF-IDF (Baseline)")
            st.caption("Dựa trên từ khóa và tần suất.")
            
        with col2:
            st.info("💡 Mô hình SBERT sẽ được so sánh ở đây sau khi bạn hoàn thành file src/recommender_sbert.py.")


# --- TAB 2: TÌM KIẾM THEO TỪ KHÓA (KEYWORD SEARCH) ---

with tab2:
    st.header("Tìm kiếm Phim theo Từ khóa")
    st.markdown("Sử dụng các từ khóa chính xác như: *'superhero action'*, *'romantic comedy'*")
    
    search_query = st.text_input("Nhập câu truy vấn của bạn:", "phim hành động có siêu anh hùng")
    
    search_k = st.slider("Số lượng kết quả tìm kiếm (Top K):", min_value=5, max_value=20, value=10, key='search_k')

    if search_query:
        # Lấy kết quả từ TF-IDF (Keyword Search)
        tfidf_search_results = tfidf_rec.search_movies(search_query, top_k=search_k, similarity_threshold=0.05)
        
        # CHỈ HIỂN THỊ CỘT TF-IDF
        col1, col2 = st.columns(2)
        
        with col1:
            display_rich_results(tfidf_search_results, "TF-IDF (Tìm kiếm Từ khóa)")

st.sidebar.markdown("---")
st.sidebar.success("✅ Hệ thống TF-IDF đã sẵn sàng để demo!")