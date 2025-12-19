import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
import plotly.express as px


# Import các hàm load model (Giả định path đã đúng)
from src.recommender_sbert import load_sbert_models
from src.recommender_tfidf import load_tfidf_models 

# 2. CẤU HÌNH ĐƯỜNG DẪN 
SBERT_MODEL_PATH = "models/sbert_model"
SBERT_EMBEDDINGS_PATH = "models/sbert_embeddings.pt"
DATA_PATH = "data/clean_movies.csv"
TFIDF_MODEL_PATH = 'models/tfidf_vectorizer.pkl'
TFIDF_MATRIX_PATH = 'models/tfidf_matrix.npy'

# Config trang
st.set_page_config(
    page_title="🎬 Hybrid Recommender Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. KHỞI TẠO STATE & LOAD MODEL

# Hàm load model đã được rút gọn và sử dụng st.cache_resource
@st.cache_resource(show_spinner=False)
def load_hybrid_recommenders_cached():
    """Load SBERT và TF-IDF recommender với cache."""
    sbert_rec, tfidf_rec = None, None
    
    # Load SBERT
    try:
        with st.spinner("🧠 Đang khởi tạo SBERT..."):
            sbert_rec = load_sbert_models(SBERT_MODEL_PATH, SBERT_EMBEDDINGS_PATH, DATA_PATH)
    except Exception as e:
        st.error(f"❌ Lỗi SBERT: {e}")

    # Load TF-IDF
    try:
        with st.spinner("📚 Đang khởi tạo TF-IDF..."):
            tfidf_rec = load_tfidf_models(TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH, DATA_PATH)
    except Exception as e:
        st.error(f"❌ Lỗi TF-IDF: {e}")

    return sbert_rec, tfidf_rec

def auto_load_model():
    """Tự động load model và cập nhật session state."""
    if 'sbert_recommender' not in st.session_state:
        with st.spinner("🔄 Đang khởi tạo Hệ thống Hybrid..."):
            sbert_rec, tfidf_rec = load_hybrid_recommenders_cached()
            
            if sbert_rec and tfidf_rec:
                st.session_state.sbert_recommender = sbert_rec
                st.session_state.tfidf_recommender = tfidf_rec
                st.session_state.movie_titles = tfidf_rec.df['title'].tolist()
                st.success(f"✅ Hệ thống Hybrid đã sẵn sàng! Dataset: {len(tfidf_rec.df):,} phim.")
            else:
                st.error("❌ Không thể khởi tạo đầy đủ hệ thống.")
                st.stop()
                
    # Đảm bảo state tồn tại sau khi load thành công
    if 'movie_titles' not in st.session_state:
        st.session_state.movie_titles = st.session_state.tfidf_recommender.df['title'].tolist()


# 4. HÀM HIỂN THỊ (Rút gọn)
def display_movie_card(movie, index, model_type='SBERT'):
    """Hiển thị thẻ phim đơn giản và hiệu quả hơn."""
    similarity = movie.get('similarity_score', 0)
    year = str(int(movie.get('year'))) if pd.notna(movie.get('year')) and str(movie.get('year')).isdigit() else 'N/A'
    rating = str(movie.get('rating', 'N/A'))
    
    # Đặt màu sắc (CSS inline style)
    if model_type == 'SBERT':
        color = "#10b981" # Green
        emoji = "🧠"
    else: # TF-IDF
        color = "#6366f1" # Indigo
        emoji = "🎯"

    # Sử dụng st.columns và Markdown/HTML đơn giản
    cols = st.columns([1, 4])
    
    with cols[0]:
        poster_url = movie.get('poster', '')
        # Hiển thị ảnh hoặc placeholder
        st.image(poster_url, width=100)
    
    with cols[1]:
        st.markdown(f"**{emoji} {index}. {movie['title']}** <span style='color:gray; font-size:14px;'>({year})</span> <span style='color:gray; font-size:14px;'>| Rating: {rating}</span>", unsafe_allow_html=True)
        st.markdown(f"**Score:** <span style='color:{color}; font-weight:bold;'>{similarity:.4f}</span> | **Model:** {model_type}", unsafe_allow_html=True)
        st.caption(f"🎭 {movie.get('genre', 'N/A')}")
        
        if movie.get('description'):
            with st.expander("📖 Mô tả"):
                st.write(movie['description'])
        
        if movie.get('review'):
            with st.expander("📝 Review"):
                st.write(movie['review'])
        
    st.markdown("---") # Phân cách các thẻ


# 5. CHỨC NĂNG SEMANTIC SEARCH
def semantic_search_section():
    """Phần test Semantic Search - SỬ DỤNG SBERT"""
    st.header("🔍 Semantic Search (SBERT)")
    st.markdown("Tìm kiếm phim dựa trên **ý nghĩa** câu truy vấn.")
    
    sbert_rec = st.session_state.sbert_recommender

    with st.form("semantic_form"):
        query = st.text_area("✨ Nhập câu truy vấn:", placeholder="VD: 'phim về tình yêu tuổi học trò lãng mạn'...", height=70, key="search_query")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Số kết quả:", 1, 20, 10, key="search_top_k")
        with col2:
            threshold = st.slider("Ngưỡng similarity:", 0.0, 1.0, 0.3, 0.05, key="search_threshold")
        
        submitted = st.form_submit_button("🚀 Chạy Semantic Search (SBERT)", type="primary", use_container_width=True)

    if submitted and query:
        start_time = time.time()
        # GỌI HÀM SBERT SEARCH
        results = sbert_rec.search_movies(query, top_k=top_k, similarity_threshold=threshold)
        search_time = time.time() - start_time
        
        st.subheader(f"📊 Kết quả tìm kiếm (SBERT)")
        
        if results:
            scores = [movie['similarity_score'] for movie in results]
            max_score = np.max(scores)
            avg_score = np.mean(scores)
            
            st.info(f"Tổng: **{len(results)}** | Max Score: **{max_score:.3f}** | TB Score: **{avg_score:.3f}** | Thời gian: **{search_time:.3f}s**")

            for i, movie in enumerate(results):
                display_movie_card(movie, i+1, model_type='SBERT') 
        else:
            st.warning("🤷 Không tìm thấy kết quả nào phù hợp")
            
# 6. CHỨC NĂNG MOVIE RECOMMENDATION
def movie_recommendation_section():
    """Phần test Movie Recommendation - SỬ DỤNG TF-IDF"""
    st.header("🎯 Movie Recommendation (TF-IDF)")
    st.markdown("Tìm phim **tương tự** dựa trên **từ khóa** (Cosine Similarity).")
    
    tfidf_rec = st.session_state.tfidf_recommender
    
    # Rút ngắn danh sách phim trong selectbox
    #movie_list_to_show = st.session_state.movie_titles[:200]
    movie_list_to_show = st.session_state.movie_titles
    
    with st.form("tfidf_form"):
        selected_movie = st.selectbox(
            "🎬 Chọn một phim gốc:",
            movie_list_to_show,
            key="movie_select_tfidf"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Số phim tương tự:", 1, 15, 8, key="rec_top_k_tfidf")
        with col2:
            min_similarity = st.slider("Similarity tối thiểu:", 0.0, 1.0, 0.15, 0.05, key="min_sim_tfidf")
        
        submitted = st.form_submit_button("🎬 Tìm phim tương tự (TF-IDF)", type="secondary", use_container_width=True)

    if submitted and selected_movie:
        with st.spinner(f"🔍 Đang tìm phim tương tự..."):
            start_time = time.time()
            # GỌI HÀM TF-IDF GET SIMILAR MOVIES
            similar_movies = tfidf_rec.get_similar_movies(selected_movie, top_k=top_k)
            search_time = time.time() - start_time
            
            # Filter by minimum similarity
            filtered_movies = [movie for movie in similar_movies if movie['similarity_score'] >= min_similarity]
            
            st.subheader(f"📊 Phim tương tự '{selected_movie}' (TF-IDF)")
            
            if filtered_movies:
                scores = [movie['similarity_score'] for movie in filtered_movies]
                avg_score = np.mean(scores)
                
                st.info(f"Tổng: **{len(filtered_movies)}** | Độ tương đồng TB: **{avg_score:.3f}** | Thời gian: **{search_time:.3f}s**")

                for i, movie in enumerate(filtered_movies):
                    display_movie_card(movie, i+1, model_type='TF-IDF') 
            else:
                st.warning(f"🤷 Không tìm thấy phim nào có similarity >= {min_similarity}")


# 7. CHỨC NĂNG CHÍNH
def main():
    """Main function"""
    st.title("🎬 Hybrid Movie Recommender System (SBERT + TF-IDF)")
    st.markdown("Hệ thống kết hợp: **SBERT** cho tìm kiếm ngữ nghĩa, **TF-IDF** cho gợi ý tương tự theo từ khóa.")
    
    # TỰ ĐỘNG LOAD MODEL KHI APP CHẠY
    auto_load_model()
    
    # Sidebar
    st.sidebar.title("🎛️ System Status")
    if 'tfidf_recommender' in st.session_state:
        tfidf_rec = st.session_state.tfidf_recommender
        sbert_rec = st.session_state.sbert_recommender
        st.sidebar.success("**✅ Status:** Hybrid Ready")
        st.sidebar.markdown(f"**🎬 Phim:** {len(tfidf_rec.df):,}")
        st.sidebar.markdown(f"**📐 SBERT Embeddings:** {sbert_rec.sbert_embeddings.shape[1]}D")
        st.sidebar.markdown(f"**📚 TF-IDF Matrix:** {tfidf_rec.tfidf_matrix.shape[1]} features")
    else:
        st.sidebar.warning("**🔄 Status:** Đang khởi tạo...")

    # Main content tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔍 Semantic Search (SBERT)", "🎯 Movie Recommendation (TF-IDF)"])
    
    with tab1:
        semantic_search_section()
    
    with tab2:
        movie_recommendation_section()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 14px;'>
    🧠 SBERT + 🎯 TF-IDF 
    <br>
    <a href='https://github.com/mv-thuyen2004/project_semantic_movie.git' target='_blank' 
       style='color: #6366f1; text-decoration: none; font-weight: bold;'>
       ⭐️ Mã nguồn trên GitHub
    </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()