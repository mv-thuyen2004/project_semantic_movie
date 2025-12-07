# app_test.py
"""
🧠 Streamlit Hybrid Test Interface:
- Semantic Search (SBERT): Dựa trên ý nghĩa câu truy vấn (từ test_sbert_streamlit.py).
- Movie Recommendation (TF-IDF): Dựa trên độ tương đồng từ khóa của phim (từ tfidf.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
import plotly.express as px

# 1. THIẾT LẬP ĐƯỜNG DẪN GỐC (PROJECT ROOT)
# Thêm path để import từ src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import các hàm load model
from src.recommender_sbert import load_sbert_models
from src.recommender_tfidf import load_tfidf_models # Cần import TF-IDF

# Config trang
st.set_page_config(
    page_title="🎬 Hybrid Recommender Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ĐƯỜNG DẪN 

SBERT_MODEL_PATH = "../models/sbert_model"
SBERT_EMBEDDINGS_PATH = "../models/sbert_embeddings.pt"
DATA_PATH = "../data/clean_movies.csv"
TFIDF_MODEL_PATH = '../models/tfidf_vectorizer.pkl'
TFIDF_MATRIX_PATH =  '../models/tfidf_matrix.npy'


def initialize_session_state():
    """Khởi tạo session state"""
    if 'sbert_loaded' not in st.session_state:
        st.session_state.sbert_loaded = False
    if 'sbert_recommender' not in st.session_state:
        st.session_state.sbert_recommender = None
    if 'tfidf_recommender' not in st.session_state: # Thêm state cho TF-IDF
        st.session_state.tfidf_recommender = None
    if 'test_history' not in st.session_state:
        st.session_state.test_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []

# 3. HÀM LOAD MODEL (Sử dụng cache_resource cho cả hai)
@st.cache_resource(show_spinner=False)
def load_hybrid_recommenders_cached():
    """Load SBERT và TF-IDF recommender với cache - chỉ chạy 1 lần"""
    sbert_rec = None
    tfidf_rec = None
    
    # --- Load SBERT ---
    try:
        with st.spinner("🧠 Đang khởi tạo SBERT model (Semantic Search)..."):
            sbert_rec = load_sbert_models(
                SBERT_MODEL_PATH,
                SBERT_EMBEDDINGS_PATH,
                DATA_PATH
            )
    except Exception as e:
        st.error(f"❌ Lỗi khi load SBERT: {e}")

    # --- Load TF-IDF ---
    try:
        with st.spinner("📚 Đang khởi tạo TF-IDF model (Movie Recommendation)..."):
            # Sử dụng DATA_PATH chung cho cả hai mô hình (dữ liệu phim)
            tfidf_rec = load_tfidf_models(TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH, DATA_PATH)
    except Exception as e:
        st.error(f"❌ Lỗi khi load TF-IDF: {e}")

    return sbert_rec, tfidf_rec

def auto_load_model():
    """Tự động load model khi app khởi chạy"""
    if not st.session_state.sbert_loaded:
        # Hiển thị loading indicator
        with st.spinner("🔄 Đang khởi tạo Hệ thống Hybrid..."):
            sbert_rec, tfidf_rec = load_hybrid_recommenders_cached()
            
            if sbert_rec and tfidf_rec:
                st.session_state.sbert_loaded = True
                st.session_state.sbert_recommender = sbert_rec
                st.session_state.tfidf_recommender = tfidf_rec
                st.success("✅ Hệ thống Hybrid đã sẵn sàng (SBERT + TF-IDF)!")
                
                # Hiển thị thông tin model
                st.info(f"📊 Dataset: {len(sbert_rec.df):,} phim")
                st.info(f"📐 SBERT Embeddings: {sbert_rec.sbert_embeddings.shape}")
            else:
                st.error("❌ Không thể khởi tạo đầy đủ hệ thống. Vui lòng kiểm tra đường dẫn file.")
                st.stop()  # Dừng app nếu không load được model

# 4. HÀM HIỂN THỊ (Giữ nguyên từ test_sbert_streamlit.py)
def display_movie_card(movie, index, model_type='SBERT'):
    """Hiển thị thẻ phim với design đẹp"""
    similarity = movie.get('similarity_score', 0)
    
    # Màu sắc và emoji dựa trên model và similarity
    if model_type == 'SBERT':
        color_scheme = {
            'high': "#10b981", 'mid': "#f59e0b", 'low': "#ef4444", 'main': "#0f172a", 'level': "Ngữ Nghĩa"
        }
    else: # TF-IDF
        color_scheme = {
            'high': "#6366f1", 'mid': "#a5b4fc", 'low': "#3730a3", 'main': "#1e1b4b", 'level': "Từ Khóa"
        }

    if similarity > 0.7:
        color = color_scheme['high']
        emoji = "🟢"
        level = "Rất cao"
    elif similarity > 0.5:
        color = color_scheme['high']
        emoji = "🟡"
        level = "Cao"
    elif similarity > 0.3:
        color = color_scheme['mid']
        emoji = "🟠"
        level = "Trung bình"
    else:
        color = color_scheme['low']
        emoji = "🔴"
        level = "Thấp"
    
    with st.container():
        poster_url = movie.get('poster', '')
        # FIX LỖI: Đảm bảo year là string an toàn
        year = str(int(movie.get('year'))) if pd.notna(movie.get('year')) and str(movie.get('year')).isdigit() else 'N/A'

        img_html = f"<img src='{poster_url}' alt='poster' style='width:120px; height:auto; border-radius:6px; margin-right:12px;'/>" if poster_url else ""

        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #1e293b, #0f172a);">
            <div style="display: flex; gap: 12px; align-items: start;">
                <div style='flex: 0 0 120px;'>
                    {img_html}
                </div>
                <div style="flex: 1;">
                    <h3 style="margin: 0; color: white;">{emoji} {index}. {movie['title']} <span style='color:#94a3b8; font-size:14px;'>({year})</span></h3>
                    <p style="margin: 5px 0; color: #94a3b8;">🎭 {movie.get('genre', 'N/A')}</p>
                    <p style="margin: 0; color: #6b7280; font-size: 12px;">Mô hình: {model_type}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: bold; color: {color};">{similarity:.3f}</div>
                    <div style="font-size: 12px; color: {color};">{level} ({color_scheme['level']})</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if movie.get('description'):
            with st.expander("📖 Xem mô tả"):
                st.write(movie['description'])


# 5. CHỨC NĂNG SEMANTIC SEARCH (Giữ nguyên từ test_sbert_streamlit.py)
def semantic_search_section():
    """Phần test Semantic Search - SỬ DỤNG SBERT"""
    st.header("🔍 Semantic Search (SBERT)")
    st.markdown("Tìm kiếm phim dựa trên **ý nghĩa** câu truy vấn sử dụng SBERT embeddings")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_area(
            "✨ Nhập câu truy vấn:",
            placeholder="VD: 'phim về tình yêu tuổi học trò lãng mạn'...",
            height=100,
            key="search_query"
        )
    
    with col2:
        top_k = st.slider("Số kết quả:", 1, 20, 10, key="search_top_k")
    
    with col3:
        threshold = st.slider("Ngưỡng similarity:", 0.0, 1.0, 0.3, 0.05, key="search_threshold")
    
    if st.button("🚀 Chạy Semantic Search (SBERT)", type="primary", use_container_width=True) and query:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🧠 Đang phân tích ngữ nghĩa..."):
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            start_time = time.time()
            # GỌI HÀM SBERT SEARCH
            results = st.session_state.sbert_recommender.search_movies(
                query, top_k=top_k, similarity_threshold=threshold
            )
            search_time = time.time() - start_time
        
        progress_bar.empty()
        status_text.empty()
        
        st.subheader(f"📊 Kết quả tìm kiếm (SBERT)")
        
        if results:
            scores = [movie['similarity_score'] for movie in results]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📈 Tổng kết quả", len(results))
            col2.metric("🎯 Điểm cao nhất", f"{max_score:.3f}")
            col3.metric("📊 Điểm trung bình", f"{avg_score:.3f}")
            col4.metric("⏱️ Thời gian", f"{search_time:.3f}s")
            
            st.markdown("---")
            st.subheader("🎬 Danh sách phim tìm được (SBERT)")
            
            for i, movie in enumerate(results):
                display_movie_card(movie, i+1, model_type='SBERT') # Đảm bảo model_type là SBERT
            
            st.session_state.current_results = results
        else:
            st.warning("🤷 Không tìm thấy kết quả nào phù hợp")
        
        # Save to history (Nếu cần)
        # ... (phần code lưu lịch sử giữ nguyên)


# 6. CHỨC NĂNG MOVIE RECOMMENDATION (Điều chỉnh để SỬ DỤNG TF-IDF)
def movie_recommendation_section():
    """Phần test Movie Recommendation - SỬ DỤNG TF-IDF"""
    st.header("🎯 Movie Recommendation (TF-IDF)")
    st.markdown("Tìm phim **tương tự** dựa trên **từ khóa** (cosine similarity trên TF-IDF)")
    
    # Get movie list
    movie_titles = st.session_state.tfidf_recommender.df['title'].tolist() # Lấy từ df của TF-IDF
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "🎬 Chọn một phim:",
            movie_titles[:200], # Giới hạn 200 phim đầu cho tốc độ load
            key="movie_select_tfidf" # Đổi key để tránh trùng với SBERT (nếu có)
        )
        
        # Show selected movie info (Lấy thông tin từ df của TF-IDF)
        if selected_movie:
            recommender = st.session_state.tfidf_recommender
            movie_idx = recommender.movie_indices_map[selected_movie]
            movie_info = recommender.df.iloc[movie_idx]
            
            st.markdown("**📋 Thông tin phim được chọn:**")
            
            poster = movie_info.get('poster', '')
            year = str(int(movie_info.get('year'))) if pd.notna(movie_info.get('year')) and str(movie_info.get('year')).isdigit() else 'N/A'
            
            poster_html = f"<img src='{poster}' style='width:120px; height:auto; border-radius:6px; margin-right:12px;'/>" if poster else ""

            st.markdown(f"""
            <div style="border: 2px solid #6366f1; border-radius: 10px; padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #3730a3, #1e1b4b);">
                <div style='display:flex; gap:12px; align-items:center;'>
                    <div style='flex:0 0 120px;'>
                        {poster_html}
                    </div>
                    <div style='flex:1;'>
                        <h4 style="margin: 0; color: white;">{selected_movie} <span style='color:#a5b4fc; font-size:14px;'>({year})</span></h4>
                        <p style="margin: 5px 0; color: #a5b4fc;">🎭 {movie_info['genre']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'description' in movie_info and pd.notna(movie_info['description']):
                with st.expander("📖 Xem mô tả phim"):
                    st.write(movie_info['description'])
    
    with col2:
        top_k = st.slider("Số phim tương tự:", 1, 15, 8, key="rec_top_k_tfidf")
        min_similarity = st.slider("Similarity tối thiểu:", 0.0, 1.0, 0.1, 0.05, key="min_sim_tfidf") # Giảm ngưỡng cho TF-IDF
    
    if st.button("🎬 Tìm phim tương tự (TF-IDF)", type="secondary", use_container_width=True) and selected_movie:
        with st.spinner(f"🔍 Đang tìm phim tương tự bằng TF-IDF..."):
            start_time = time.time()
            # GỌI HÀM TF-IDF GET SIMILAR MOVIES
            similar_movies = st.session_state.tfidf_recommender.get_similar_movies(
                selected_movie, top_k=top_k
            )
            search_time = time.time() - start_time
        
        # Filter by minimum similarity
        filtered_movies = [movie for movie in similar_movies if movie['similarity_score'] >= min_similarity]
        
        st.subheader(f"📊 Phim tương tự '{selected_movie}' (TF-IDF)")
        
        if filtered_movies:
            scores = [movie['similarity_score'] for movie in filtered_movies]
            avg_score = np.mean(scores)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Tổng phim tương tự", len(filtered_movies))
            col2.metric("📊 Độ tương đồng TB", f"{avg_score:.3f}")
            col3.metric("⏱️ Thời gian", f"{search_time:.3f}s")
            
            st.markdown("---")
            st.subheader("📺 Danh sách phim tương tự (TF-IDF)")
            
            for i, movie in enumerate(filtered_movies):
                display_movie_card(movie, i+1, model_type='TF-IDF') # Đảm bảo model_type là TF-IDF
            
        else:
            st.warning(f"🤷 Không tìm thấy phim nào có similarity >= {min_similarity}")
        
        # Save to history (Nếu cần)
        # ... (phần code lưu lịch sử giữ nguyên)


def sidebar_info():
    """Hiển thị thông tin ở sidebar"""
    st.sidebar.title("🎛️ System Info")
    
    if st.session_state.sbert_loaded:
        st.sidebar.success("**✅ Status:** Hybrid Ready")
        sbert_rec = st.session_state.sbert_recommender
        tfidf_rec = st.session_state.tfidf_recommender
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🧠 SBERT Info (Semantic)**")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("🎬 Phim", f"{len(sbert_rec.df):,}")
        col2.metric("📐 Embeddings", f"{sbert_rec.sbert_embeddings.shape[1]}D")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**📚 TF-IDF Info (Keyword)**")
        st.sidebar.caption(f"Kích thước ma trận: {tfidf_rec.tfidf_matrix.shape}")
        
    else:
        st.sidebar.warning("**🔄 Status:** Đang khởi tạo...")

def main():
    """Main function"""
    # Header
    st.title("🎬 Hybrid Movie Recommender System (SBERT + TF-IDF)")
    st.markdown("**Hệ thống kết hợp:** SBERT cho tìm kiếm ngữ nghĩa, TF-IDF cho gợi ý tương tự")
    
    # Khởi tạo session state
    initialize_session_state()
    
    # TỰ ĐỘNG LOAD MODEL KHI APP CHẠY
    auto_load_model()
    
    # Hiển thị sidebar info
    sidebar_info()
    
    # Main content tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔍 Semantic Search (SBERT)", "🎯 Movie Recommendation (TF-IDF)"])
    
    with tab1:
        # Giao diện SBERT Search
        semantic_search_section()
    
    with tab2:
        # Giao diện TF-IDF Recommendation
        movie_recommendation_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280;'>"
        "🧠 SBERT (Search) + 📚 TF-IDF (Recommend) • Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()