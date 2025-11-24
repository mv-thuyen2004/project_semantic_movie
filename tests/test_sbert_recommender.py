# tests/test_sbert_streamlit.py
"""
🧠 Streamlit Test Interface cho SBERT Recommender
Phiên bản tự động load model khi khởi chạy
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
import plotly.express as px

# Thêm path để import từ src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Config trang
st.set_page_config(
    page_title="🧠 SBERT Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ĐƯỜNG DẪN TUYỆT ĐỐI
SBERT_MODEL_PATH = "D:/datamining/project_semantic_movie/models/sbert_model"
SBERT_EMBEDDINGS_PATH = "D:/datamining/project_semantic_movie/models/sbert_embeddings.pt"
DATA_PATH = "D:/datamining/project_semantic_movie/data/clean_movies.csv"

def initialize_session_state():
    """Khởi tạo session state"""
    if 'sbert_loaded' not in st.session_state:
        st.session_state.sbert_loaded = False
    if 'sbert_recommender' not in st.session_state:
        st.session_state.sbert_recommender = None
    if 'test_history' not in st.session_state:
        st.session_state.test_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []

@st.cache_resource(show_spinner=False)
def load_sbert_recommender_cached():
    """Load SBERT recommender với cache - chỉ chạy 1 lần"""
    try:
        from src.recommender_sbert import load_sbert_models
        
        with st.spinner("🧠 Đang khởi tạo SBERT model..."):
            recommender = load_sbert_models(
                SBERT_MODEL_PATH,
                SBERT_EMBEDDINGS_PATH,
                DATA_PATH
            )
        
        return recommender
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        return None

def auto_load_model():
    """Tự động load model khi app khởi chạy"""
    if not st.session_state.sbert_loaded:
        # Hiển thị loading indicator
        with st.spinner("🔄 Đang khởi tạo hệ thống SBERT..."):
            recommender = load_sbert_recommender_cached()
            
            if recommender:
                st.session_state.sbert_loaded = True
                st.session_state.sbert_recommender = recommender
                st.success("✅ Hệ thống SBERT đã sẵn sàng!")
                
                # Hiển thị thông tin model
                st.info(f"📊 Dataset: {len(recommender.df):,} phim")
                st.info(f"📐 Embeddings: {recommender.sbert_embeddings.shape}")
            else:
                st.error("❌ Không thể khởi tạo hệ thống SBERT")
                st.stop()  # Dừng app nếu không load được model

def display_movie_card(movie, index):
    """Hiển thị thẻ phim với design đẹp"""
    similarity = movie.get('similarity_score', 0)
    
    # Màu sắc và emoji dựa trên similarity
    if similarity > 0.7:
        color = "#10b981"
        emoji = "🟢"
        level = "Rất cao"
    elif similarity > 0.5:
        color = "#f59e0b"
        emoji = "🟡"
        level = "Cao"
    elif similarity > 0.3:
        color = "#f97316"
        emoji = "🟠"
        level = "Trung bình"
    else:
        color = "#ef4444"
        emoji = "🔴"
        level = "Thấp"
    
    with st.container():
        # Prepare poster and year
        poster_url = movie.get('poster', '')
        year = movie.get('year', 'N/A')

        # Tạo card với border, hiển thị poster nếu có
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
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: bold; color: {color};">{similarity:.3f}</div>
                    <div style="font-size: 12px; color: {color};">{level}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if movie.get('description'):
            with st.expander("📖 Xem mô tả"):
                st.write(movie['description'])

def semantic_search_section():
    """Phần test Semantic Search"""
    st.header("🔍 Semantic Search")
    st.markdown("Tìm kiếm phim dựa trên **ý nghĩa** câu truy vấn sử dụng SBERT embeddings")
    
    # Input section
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_area(
            "✨ Nhập câu truy vấn:",
            placeholder="VD: 'phim về tình yêu tuổi học trò lãng mạn' hoặc 'phim hành động có cảnh đánh nhau đẹp mắt'...",
            height=100,
            key="search_query"
        )
    
    with col2:
        top_k = st.slider("Số kết quả:", 1, 20, 10, key="search_top_k")
    
    with col3:
        threshold = st.slider("Ngưỡng similarity:", 0.0, 1.0, 0.3, 0.05, key="search_threshold")
    
    if st.button("🚀 Chạy Semantic Search", type="primary", use_container_width=True) and query:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🧠 Đang phân tích ngữ nghĩa..."):
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            start_time = time.time()
            results = st.session_state.sbert_recommender.search_movies(
                query, top_k=top_k, similarity_threshold=threshold
            )
            search_time = time.time() - start_time
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.subheader(f"📊 Kết quả tìm kiếm")
        
        if results:
            # Statistics
            scores = [movie['similarity_score'] for movie in results]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📈 Tổng kết quả", len(results))
            col2.metric("🎯 Điểm cao nhất", f"{max_score:.3f}")
            col3.metric("📊 Điểm trung bình", f"{avg_score:.3f}")
            col4.metric("⏱️ Thời gian", f"{search_time:.3f}s")
            
            st.markdown("---")
            st.subheader("🎬 Danh sách phim tìm được")
            
            # Movie results
            for i, movie in enumerate(results):
                display_movie_card(movie, i+1)
            
            st.session_state.current_results = results
            
        else:
            st.warning("🤷 Không tìm thấy kết quả nào phù hợp")
        
        # Save to history
        st.session_state.test_history.append({
            'timestamp': datetime.now(),
            'type': 'semantic_search',
            'query': query,
            'results_count': len(results),
            'search_time': search_time,
            'avg_similarity': avg_score if results else 0
        })

def movie_recommendation_section():
    """Phần test Movie Recommendation"""
    st.header("🎯 Movie Recommendation")
    st.markdown("Tìm phim **tương tự** dựa trên embedding ngữ nghĩa của phim được chọn")
    
    # Get movie list
    movie_titles = st.session_state.sbert_recommender.df['title'].tolist()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "🎬 Chọn một phim:",
            movie_titles[:200],
            key="movie_select"
        )
        
        # Show selected movie info
        if selected_movie:
            movie_idx = st.session_state.sbert_recommender.movie_indices_map[selected_movie]
            movie_info = st.session_state.sbert_recommender.df.iloc[movie_idx]
            
            st.markdown("**📋 Thông tin phim được chọn:**")
            
            poster = movie_info.get('poster', '')
            year = movie_info.get('year', 'N/A')
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
        top_k = st.slider("Số phim tương tự:", 1, 15, 8, key="rec_top_k")
        min_similarity = st.slider("Similarity tối thiểu:", 0.0, 1.0, 0.4, 0.05)
    
    if st.button("🎬 Tìm phim tương tự", type="primary", use_container_width=True) and selected_movie:
        with st.spinner(f"🔍 Đang tìm phim tương tự..."):
            start_time = time.time()
            similar_movies = st.session_state.sbert_recommender.get_similar_movies(
                selected_movie, top_k=top_k
            )
            search_time = time.time() - start_time
        
        # Filter by minimum similarity
        filtered_movies = [movie for movie in similar_movies if movie['similarity_score'] >= min_similarity]
        
        st.subheader(f"📊 Phim tương tự '{selected_movie}'")
        
        if filtered_movies:
            scores = [movie['similarity_score'] for movie in filtered_movies]
            avg_score = np.mean(scores)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Tổng phim tương tự", len(filtered_movies))
            col2.metric("📊 Độ tương đồng TB", f"{avg_score:.3f}")
            col3.metric("⏱️ Thời gian", f"{search_time:.3f}s")
            
            st.markdown("---")
            st.subheader("📺 Danh sách phim tương tự")
            
            for i, movie in enumerate(filtered_movies):
                display_movie_card(movie, i+1)
            
        else:
            st.warning(f"🤷 Không tìm thấy phim nào có similarity >= {min_similarity}")
        
        st.session_state.test_history.append({
            'timestamp': datetime.now(),
            'type': 'movie_recommendation', 
            'source_movie': selected_movie,
            'results_count': len(filtered_movies),
            'search_time': search_time,
            'avg_similarity': avg_score if filtered_movies else 0
        })

def sidebar_info():
    """Hiển thị thông tin ở sidebar"""
    st.sidebar.title("🎛️ System Info")
    
    if st.session_state.sbert_loaded:
        st.sidebar.success("**✅ Status:** Ready")
        recommender = st.session_state.sbert_recommender
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("🎬 Phim", f"{len(recommender.df):,}")
        col2.metric("📐 Embeddings", f"{recommender.sbert_embeddings.shape[1]}D")
        
    else:
        st.sidebar.warning("**🔄 Status:** Đang khởi tạo...")

def main():
    """Main function"""
    # Header
    st.title("🎬 SBERT Movie Recommender System")
    st.markdown("**Hệ thống gợi ý phim thông minh** sử dụng SBERT embeddings")
    
    # Khởi tạo session state
    initialize_session_state()
    
    # TỰ ĐỘNG LOAD MODEL KHI APP CHẠY
    auto_load_model()
    
    # Hiển thị sidebar info
    sidebar_info()
    
    # Main content tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔍 Semantic Search", "🎯 Movie Recommendation"])
    
    with tab1:
        semantic_search_section()
    
    with tab2:
        movie_recommendation_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280;'>"
        "🎬 Powered by SBERT • Built with Streamlit • "
        "<a href='https://github.com/mv-thuyen2004/project_semantic_movie.git' target='_blank'>GitHub</a>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()