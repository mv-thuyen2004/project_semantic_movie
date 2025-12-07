# test_tf.py
import streamlit as st
import pandas as pd
import os
import sys

# Thêm đường dẫn để import được module của bạn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tf_idf import load_tfidf_models, TFIDFRecommender

# ==================== CÀI ĐẶT ĐƯỜNG DẪN MODEL (CHỈNH THEO BẠN) ====================
TFIDF_MODEL_PATH = "models/tfidf_vectorizer.pkl"        # Đường dẫn đến file vectorizer
TFIDF_MATRIX_PATH = "models/tfidf_matrix.npy"              # hoặc .joblib
DATA_CSV_PATH = "data/clean_movies.csv"                  # File CSV chứa dữ liệu phim

# ==================== LOAD MODEL (CHỈ LOAD 1 LẦN) ====================
@st.cache_resource
def load_recommender():
    with st.spinner("Đang tải mô hình TF-IDF gợi ý phim tương tự... (chỉ tải 1 lần)"):
        recommender = load_tfidf_models(
            tfidf_path=TFIDF_MODEL_PATH,
            matrix_path=TFIDF_MATRIX_PATH,
            data_path=DATA_CSV_PATH
        )
    return recommender

recommender = load_recommender()

if recommender is None:
    st.error("Không load được mô hình TF-IDF. Kiểm tra lại đường dẫn file!")
    st.stop()

# ==================== GIAO DIỆN STREAMLIT ====================
st.set_page_config(page_title="TF-IDF Movie Recommender", layout="centered")
st.title("Phim Tương Tự (More Like This)")
st.caption("Dùng TF-IDF + Metadata (genre, cast, director, description) - Đã tối ưu trọng số cho data Việt Nam")

st.sidebar.header("Chọn phim để tìm tương tự")
all_titles = recommender.df['title'].dropna().unique().tolist()
all_titles = sorted(all_titles)

# Tìm kiếm phim theo từ khóa
search_query = st.sidebar.text_input("Tìm phim nhanh:", placeholder="Nhập tên phim...")
if search_query:
    matched = [t for t in all_titles if search_query.lower() in t.lower()]
    selected_title = st.sidebar.selectbox(f"Kết quả tìm kiếm ({len(matched)} phim):", matched)
else:
    selected_title = st.sidebar.selectbox("Chọn một bộ phim:", all_titles)

top_k = st.sidebar.slider("Số lượng phim gợi ý:", 5, 20, 10)

if st.sidebar.button("Tìm phim tương tự", type="primary"):
    with st.spinner(f"Đang tìm phim tương tự cho **{selected_title}**..."):
        try:
            # Dùng get_similar_movies (theo title)
            results = recommender.get_similar_movies(
                movie_title=selected_title,
                top_k=top_k,
                exclude_self=True
            )

            if not results:
                st.warning(f"Không tìm thấy phim tương tự cho: {selected_title}")
            else:
                st.success(f"Tìm thấy {len(results)} phim tương tự cho **{selected_title}**")

                cols = st.columns(3)
                for i, movie in enumerate(results):
                    with cols[i % 3]:
                        st.markdown(f"### {i+1}. **{movie['title']}**")
                        if pd.notna(movie.get('year')):
                            st.caption(f"**Năm:** {int(movie['year'])} | **Điểm tương đồng:** {movie['similarity_score']:.3f}")

                        if movie.get('poster') and movie['poster'] != 'N/A':
                            st.image(movie['poster'], use_column_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x450.png?text=No+Poster", use_column_width=True)

                        genre = movie.get('genre', 'N/A')
                        if genre and genre != 'N/A':
                            st.caption(f"**Thể loại:** {genre}")

                        with st.expander("Xem mô tả"):
                            st.write(movie.get('description', 'Không có mô tả'))

        except Exception as e:
            st.error(f"Lỗi khi gợi ý: {e}")
            st.write("Chi tiết lỗi:", e)

