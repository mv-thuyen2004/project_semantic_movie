import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
import plotly.express as px

# 1. THIẾT LẬP ĐƯỜNG DẪN GỐC & IMPORT
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import các hàm load model
from src.recommender_sbert import load_sbert_models
from src.recommender_tfidf import load_tfidf_models 

# 2. CẤU HÌNH ĐƯỜNG DẪN (sử dụng project_root)
SBERT_MODEL_PATH = os.path.join(project_root, "models/sbert_model")
SBERT_EMBEDDINGS_PATH = os.path.join(project_root, "models/sbert_embeddings.pt")
DATA_PATH = os.path.join(project_root, "data/clean_movies.csv")
TFIDF_MODEL_PATH = os.path.join(project_root, "models/tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(project_root, "models/tfidf_matrix.npy")

# Config trang
st.set_page_config(
    page_title="🎬 Hybrid Recommender Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. HÀM HIỂN THỊ MOVIE CARD
def display_movie_card(movie, index, model_type='SBERT'):
    """Hiển thị thẻ phim đơn giản và hiệu quả"""
    similarity = movie.get('similarity_score', 0)
    year = str(int(movie.get('year'))) if pd.notna(movie.get('year')) and str(movie.get('year')).isdigit() else 'N/A'
    rating = str(movie.get('rating', 'N/A'))
    
    # Đặt màu sắc theo model type
    if model_type == 'SBERT':
        color = "#10b981"  # Green
        emoji = "🧠"
    else:  # TF-IDF
        color = "#6366f1"  # Indigo
        emoji = "🎯"

    # Sử dụng columns để bố cục
    cols = st.columns([1, 4])
    
    with cols[0]:
        poster_url = movie.get('poster', '')
        if poster_url and poster_url != 'N/A':
            st.image(poster_url, width=100, caption=f"Rank: #{index}")
        else:
            st.image("https://via.placeholder.com/100x150/374151/FFFFFF?text=No+Poster", 
                    width=100, caption=f"Rank: #{index}")
    
    with cols[1]:
        # Tiêu đề phim
        st.markdown(f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-size: 18px; font-weight: bold;'>{emoji} {index}. {movie['title']}</span>
            <span style='color: #6b7280; font-size: 14px; margin-left: 10px;'>({year})</span>
            <span style='color: #6b7280; font-size: 14px; margin-left: 10px;'>⭐ {rating}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Thông tin similarity và model
        st.markdown(f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-weight: bold;'>Score:</span> 
            <span style='color:{color}; font-weight:bold; margin-right: 15px;'>{similarity:.4f}</span>
            <span style='font-weight: bold;'>Model:</span> {model_type}
        </div>
        """, unsafe_allow_html=True)
        
        # Thể loại
        genres = movie.get('genre', 'N/A')
        if genres != 'N/A':
            st.caption(f"🎭 {genres}")
        
        # Mô tả (expandable)
        description = movie.get('description', '')
        if description and description != 'N/A':
            with st.expander("📖 Xem mô tả"):
                st.write(description[:300] + "..." if len(description) > 300 else description)
        
        # Review (expandable)
        review = movie.get('review', '')
        if review and review != 'N/A':
            with st.expander("📝 Xem review"):
                st.write(review[:200] + "..." if len(review) > 200 else review)
    
    st.markdown("---")

# 4. HÀM HIỂN THỊ KẾT QUẢ CHUNG
def display_search_results(results, model_type='SBERT', query_title=None, search_time=None):
    """Hiển thị kết quả tìm kiếm/recommendation (dùng chung)"""
    if not results:
        st.warning("🤷 Không tìm thấy kết quả nào phù hợp")
        return
    
    # Tính toán metrics
    scores = [movie['similarity_score'] for movie in results]
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    
    # Tiêu đề
    if query_title:
        st.subheader(f"📊 Kết quả cho '{query_title}' ({model_type})")
    else:
        st.subheader(f"📊 Kết quả tìm kiếm ({model_type})")
    
    # Hiển thị thông tin tổng quan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng kết quả", len(results))
    with col2:
        st.metric("Điểm cao nhất", f"{max_score:.3f}")
    with col3:
        st.metric("Điểm trung bình", f"{avg_score:.3f}")
    with col4:
        if search_time:
            st.metric("Thời gian xử lý", f"{search_time:.2f}s")
    
    # Hiển thị từng phim
    for i, movie in enumerate(results):
        display_movie_card(movie, i+1, model_type)

# 5. LOAD MODEL VÀ KHỞI TẠO SESSION STATE
@st.cache_resource(show_spinner=False)
def load_models_cached():
    """Load cả hai model một lần và cache"""
    try:
        with st.spinner("🧠 Đang khởi tạo SBERT..."):
            sbert_rec = load_sbert_models(SBERT_MODEL_PATH, SBERT_EMBEDDINGS_PATH, DATA_PATH)
        
        with st.spinner("📚 Đang khởi tạo TF-IDF..."):
            tfidf_rec = load_tfidf_models(TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH, DATA_PATH)
        
        return sbert_rec, tfidf_rec
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {str(e)[:200]}")
        return None, None

def init_session_state():
    """Khởi tạo session state"""
    if 'models_loaded' not in st.session_state:
        sbert_rec, tfidf_rec = load_models_cached()
        
        if sbert_rec and tfidf_rec:
            st.session_state.update({
                'sbert_recommender': sbert_rec,
                'tfidf_recommender': tfidf_rec,
                'movie_titles': tfidf_rec.df['title'].tolist(),
                'models_loaded': True
            })
        else:
            st.error("❌ Không thể khởi tạo hệ thống. Vui lòng kiểm tra đường dẫn model.")
            st.stop()

# 6. SEMANTIC SEARCH SECTION
def semantic_search_section():
    """Phần Semantic Search với SBERT"""
    st.header("🔍 Semantic Search (SBERT)")
    st.markdown("Tìm kiếm phim dựa trên **ý nghĩa ngữ nghĩa** của câu truy vấn.")
    
    # Thông tin về SBERT
    with st.expander("ℹ️ Tìm hiểu về SBERT", expanded=False):
        st.markdown("""
        **SBERT (Sentence-BERT)** sử dụng mô hình BERT để mã hóa câu thành vector.
        - Hiểu ngữ nghĩa sâu, không chỉ từ khóa
        - Tìm phim có nội dung tương tự về ý nghĩa
        - Hiệu quả với truy vấn phức tạp, nhiều ý
        """)
    
    sbert_rec = st.session_state.sbert_recommender
    
    with st.form("semantic_form"):
        # Input query
        query = st.text_area(
            "✨ Nhập mô tả phim bạn muốn tìm:",
            placeholder="VD: 'phim về tình yêu tuổi học trò lãng mạn có cảnh mưa' hoặc 'phim hành động đánh nhau hay của Marvel'...",
            height=100
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Số kết quả tối đa:", 1, 30, 10)
        with col2:
            threshold = st.slider("Ngưỡng similarity tối thiểu:", 0.0, 1.0, 0.25, 0.05)
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Chạy Semantic Search",
            type="primary",
            use_container_width=True
        )
    
    # Xử lý khi submit
    if submitted and query:
        if not query.strip():
            st.warning("Vui lòng nhập câu truy vấn!")
            return
            
        with st.spinner("🔍 Đang phân tích ngữ nghĩa..."):
            start_time = time.time()
            
            # Gọi hàm search của SBERT
            results = sbert_rec.search_movies(
                query, 
                top_k=top_k, 
                similarity_threshold=threshold
            )
            
            search_time = time.time() - start_time
        
        # Lọc kết quả theo threshold
        filtered_results = [r for r in results if r['similarity_score'] >= threshold]
        
        # Hiển thị kết quả
        if filtered_results:
            display_search_results(
                filtered_results, 
                model_type='SBERT',
                search_time=search_time
            )
        else:
            st.warning(f"Không tìm thấy phim nào với similarity >= {threshold}")
            
            # Gợi ý
            st.info("""
            💡 **Gợi ý:**
            - Giảm ngưỡng similarity
            - Mô tả chi tiết hơn về phim bạn muốn tìm
            - Thử từ khóa khác
            """)

# 7. MOVIE RECOMMENDATION SECTION
def movie_recommendation_section():
    """Phần Movie Recommendation với TF-IDF"""
    st.header("🎯 Movie Recommendation (TF-IDF)")
    st.markdown("Tìm phim **tương tự** dựa trên **từ khóa và nội dung**.")
    
    # Thông tin về TF-IDF
    with st.expander("ℹ️ Tìm hiểu về TF-IDF", expanded=False):
        st.markdown("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)** phân tích tần suất từ.
        - Tìm phim có từ khóa và nội dung tương tự
        - Hiệu quả khi biết tên phim cụ thể
        - Dựa trên overlap của từ và cụm từ
        """)
    
    tfidf_rec = st.session_state.tfidf_recommender
    
    with st.form("tfidf_form"):
        # Movie selection
        selected_movie = st.selectbox(
            "🎬 Chọn một phim gốc:",
            st.session_state.movie_titles,
            index=0 if len(st.session_state.movie_titles) > 0 else None,
            help="Chọn phim bạn thích để tìm phim tương tự"
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Số phim tương tự:", 1, 20, 8, key="rec_top_k")
        with col2:
            min_similarity = st.slider("Similarity tối thiểu:", 0.0, 1.0, 0.15, 0.05, key="min_sim")
        
        # Submit button
        submitted = st.form_submit_button(
            "🎬 Tìm phim tương tự",
            type="secondary",
            use_container_width=True
        )
    
    # Xử lý khi submit
    if submitted and selected_movie:
        with st.spinner(f"🔍 Đang tìm phim tương tự '{selected_movie}'..."):
            start_time = time.time()
            
            # Gọi hàm get_similar_movies của TF-IDF
            similar_movies = tfidf_rec.get_similar_movies(
                selected_movie, 
                top_k=top_k
            )
            
            search_time = time.time() - start_time
        
        # Lọc theo similarity
        filtered_movies = [m for m in similar_movies if m['similarity_score'] >= min_similarity]
        
        # Hiển thị phim gốc
        st.markdown("### 🎬 Phim gốc")
        original_movie = tfidf_rec.df[tfidf_rec.df['title'] == selected_movie].iloc[0].to_dict()
        display_movie_card(original_movie, 0, model_type='ORIGINAL')
        
        # Hiển thị kết quả
        if filtered_movies:
            st.markdown(f"### 📊 Danh sách phim tương tự ({len(filtered_movies)} phim)")
            display_search_results(
                filtered_movies, 
                model_type='TF-IDF',
                query_title=selected_movie,
                search_time=search_time
            )
        else:
            st.warning(f"Không tìm thấy phim nào có similarity >= {min_similarity}")
            
            # Hiển thị tất cả kết quả không lọc
            if similar_movies:
                st.info(f"Hiển thị tất cả {len(similar_movies)} kết quả (bỏ qua ngưỡng):")
                display_search_results(
                    similar_movies,
                    model_type='TF-IDF',
                    query_title=selected_movie,
                    search_time=search_time
                )

# 8. COMPARISON SECTION (Tùy chọn mở rộng)
def comparison_section():
    """Phần so sánh kết quả giữa SBERT và TF-IDF"""
    st.header("⚖️ So sánh SBERT vs TF-IDF")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 SBERT")
        st.markdown("""
        **Ưu điểm:**
        - Hiểu ngữ nghĩa sâu
        - Tốt với truy vấn phức tạp
        - Không phụ thuộc từ khóa chính xác
        
        **Nhược điểm:**
        - Chậm hơn
        - Cần nhiều tài nguyên
        - Đòi hỏi model lớn
        """)
    
    with col2:
        st.markdown("### 🎯 TF-IDF")
        st.markdown("""
        **Ưu điểm:**
        - Nhanh, hiệu quả
        - Tốt với từ khóa cụ thể
        - Dễ triển khai
        
        **Nhược điểm:**
        - Không hiểu ngữ nghĩa
        - Phụ thuộc vào từ khóa trùng khớp
        - Bỏ qua ngữ cảnh
        """)
    
    st.markdown("""
    ### 🎬 Khi nào dùng cái nào?
    - **Dùng SBERT**: Khi bạn có mô tả ý tưởng nhưng không biết tên phim
    - **Dùng TF-IDF**: Khi bạn biết phim cụ thể và muốn tìm phim tương tự
    """)

# 9. MAIN FUNCTION
def main():
    """Main function của ứng dụng"""
    # Header
    st.title("🎬 Hybrid Movie Recommender System")
    st.markdown("""
    Hệ thống kết hợp **SBERT** (tìm kiếm ngữ nghĩa) và **TF-IDF** (gợi ý tương tự) 
    để mang lại trải nghiệm tìm phim tốt nhất.
    """)
    
    # Khởi tạo session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ System Status")
        
        if st.session_state.models_loaded:
            tfidf_rec = st.session_state.tfidf_recommender
            sbert_rec = st.session_state.sbert_recommender
            
            st.success("✅ **Status:** Hybrid Ready")
            
            # Thông tin dataset
            st.markdown("### 📊 Dataset Info")
            st.metric("Tổng số phim", f"{len(tfidf_rec.df):,}")
            
            # Thông tin model
            st.markdown("### 🧠 Model Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SBERT Dim", f"{sbert_rec.sbert_embeddings.shape[1]}D")
            with col2:
                st.metric("TF-IDF Features", f"{tfidf_rec.tfidf_matrix.shape[1]}")
            
            # Thống kê thêm
            st.markdown("### 📈 Thống kê")
            if 'year' in tfidf_rec.df.columns:
                current_year = datetime.now().year
                recent_movies = tfidf_rec.df[tfidf_rec.df['year'] >= (current_year - 5)].shape[0]
                st.metric("Phim 5 năm gần đây", recent_movies)
            
            # Refresh button
            if st.button("🔄 Clear Cache & Reload", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
        else:
            st.warning("🔄 **Status:** Đang khởi tạo...")
    
    # Main content - Tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Semantic Search (SBERT)", 
        "🎯 Movie Recommendation (TF-IDF)",
        "⚖️ So sánh Model"
    ])
    
    with tab1:
        semantic_search_section()
    
    with tab2:
        movie_recommendation_section()
    
    with tab3:
        comparison_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>🎬 <b>Hybrid Movie Recommender System</b> | 🧠 SBERT + 🎯 TF-IDF</p>
        <p>Built with ❤️ using Streamlit | Dataset: Clean Movies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug info (chỉ hiển thị khi cần)
    if st.sidebar.checkbox("Show Debug Info", False):
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.write(f"Models loaded: {st.session_state.models_loaded}")
        if 'movie_titles' in st.session_state:
            st.sidebar.write(f"Movie count: {len(st.session_state.movie_titles)}")
        st.sidebar.write(f"Python: {sys.version}")

# 10. RUN APPLICATION
if __name__ == "__main__":
    main()