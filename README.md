# 🎬 Hybrid Movie Recommender System (SBERT + TF-IDF)

Hệ thống gợi ý phim thông minh kết hợp **SBERT** (tìm kiếm ngữ nghĩa) và **TF-IDF** (gợi ý theo từ khóa).

## 🎯 Tính Năng

- 🔍 **Semantic Search**: Tìm phim từ câu mô tả tự do (ví dụ: "phim tình yêu buồn")
- 🎯 **Movie Recommendation**: Gợi ý phim tương tự dựa trên một bộ phim đã chọn
- ⚡ **Hybrid**: Kết hợp hai phương pháp để cho kết quả tốt nhất

## 🏗️ Cấu Trúc Dự Án

```
project_semantic_movie/
├── app.py                      # 🚀 Ứng dụng Streamlit
├── requirements.txt            # 📦 Dependencies
├── data/
│   ├── clean_movies.csv        # ✅ Dataset sạch
│   ├── demo_movies.csv
│   └── imdb-movies-dataset.csv
├── models/
│   ├── sbert_model/            # 🧠 SBERT pre-trained model
│   ├── sbert_embeddings.pt     # 💾 Embeddings cache
│   ├── tfidf_vectorizer.pkl    # 🔧 TF-IDF vectorizer
│   └── tfidf_matrix.npy        # 📐 TF-IDF matrix
├── src/
│   ├── preprocessing.py        # 🔨 Tiền xử lý
│   ├── recommender_sbert.py    # 🧠 SBERT logic
│   ├── recommender_tfidf.py    # 🎯 TF-IDF logic
│   └── test_*.py               # ✅ Tests
├── notebook/
│   ├── Train_SBERT.ipynb       # 🧠 Huấn luyện SBERT
│   └── Train_TFIDF.ipynb       # 📚 Huấn luyện TF-IDF
└── tests/                      # 🧪 Unit tests
```

## 🚀 Cách Sử Dụng

### 1️⃣ Cài Đặt
```bash
git clone https://github.com/mv-thuyen2004/project_semantic_movie.git
cd project_semantic_movie
pip install -r requirements.txt
```

### 2️⃣ Chạy Ứng Dụng
```bash
streamlit run app.py
```
Mở tại `http://localhost:8501`

### 3️⃣ Hai Tab Chính
- **🔍 Semantic Search**: Nhập câu truy vấn tự do
- **🎯 Movie Recommendation**: Chọn phim để tìm các phim tương tự

## 🔧 Các Module Chính

### 📄 `preprocessing.py`
Tiền xử lý dữ liệu:
- `normalize_text()` - Chuẩn hóa text (lowercase, xóa dấu, ký tự đặc biệt)
- `remove_stopwords()` - Loại bỏ từ vô nghĩa (cho TF-IDF)
- `build_similarity_text()` - Tạo text cho TF-IDF (áp dụng trọng số)
- `build_full_text()` - Tạo text cho SBERT (giữ ngữ cảnh)
- `preprocess_query()` - Tiền xử lý truy vấn người dùng

### 🧠 `recommender_sbert.py`
SBERT Semantic Search:
- `search_movies(query)` - Tìm phim từ câu mô tả
- `get_similar_movies(title)` - Tìm phim tương tự

### 🎯 `recommender_tfidf.py`
TF-IDF Recommendation:
- `search_movies(query)` - Tìm phim từ từ khóa
- `get_similar_movies(title)` - Tìm phim tương tự

### 💾 `app.py`
Giao diện Streamlit:
- Load models (SBERT + TF-IDF) với cache
- 2 tab: Semantic Search & Movie Recommendation
- Hiển thị kết quả với hình ảnh, điểm số, mô tả

## 📊 Dữ Liệu & Mô Hình

**Dataset:** IMDB Movies (~5000-10000 phim)

**SBERT Model:**
- Pre-trained Sentence-BERT (all-MiniLM-L6-v2)
- Embeddings cache: `sbert_embeddings.pt` (~300-500 MB)
- Tính toán cosine similarity

**TF-IDF Model:**
- Sklearn TfidfVectorizer (~10-50 MB)
- Sparse matrix (~100-200 MB)
- Tính toán nhanh

## 📈 Huấn Luyện Mô Hình

**Train SBERT (Train_SBERT.ipynb):**
```python
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sbert_model.encode(df['full_text'], batch_size=32, normalize_embeddings=True)
torch.save(embeddings, "models/sbert_embeddings.pt")
```

**Train TF-IDF (Train_TFIDF.ipynb):**
```python
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df['similarity_text'])
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
```

## ✅ Testing & Ví Dụ

**Chạy tests:**
```bash
python -m tests.test_sbert_recommender
python -m tests.test_tfidf_recommender
```

**Ví dụ Semantic Search:**
```
Input: "phim tâm lý về trầm cảm"
Output:
1. "Requiem for a Dream" (0.89)
2. "The Perks of Being a Wallflower" (0.87)
```

**Ví dụ TF-IDF Recommendation:**
```
Input: "The Shawshank Redemption"
Output:
1. "The Green Mile" (0.78)
2. "Forrest Gump" (0.72)
```

## 📚 Tài Liệu Tham Khảo

- [Sentence-BERT](https://www.sbert.net/)
- [TF-IDF - Scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf)
- [Streamlit](https://streamlit.io/)

## 📞 Liên Hệ

GitHub: https://github.com/mv-thuyen2004/project_semantic_movie.git

---

**Chúc bạn tìm được bộ phim yêu thích! 🎬✨**