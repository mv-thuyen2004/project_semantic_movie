import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as skpair
import joblib
import sys
import os
import scipy.sparse as sp 

# Thêm path để import các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_query 

class TFIDFRecommender:
    """
    Lớp gợi ý phim sử dụng TF-IDF (Memory Safe - Dynamic 1×N Calculation)
    """
    
    def __init__(self, tfidf_model, tfidf_matrix, df):
        self.tfidf_model = tfidf_model
        self.tfidf_matrix = tfidf_matrix  
        self.df = df
        
        # Tạo mapping để tra cứu nhanh title -> index
        self.movie_indices_map = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        print("✅ TF-IDF Recommender khởi tạo thành công (Memory Safe Mode)")
    
    
    def search_movies(self, query, top_k=10, similarity_threshold=0.1):
        """
        Tìm kiếm phim dựa trên query text (Keyword Search) - Dynamic 1×N
        """
        processed_query = preprocess_query(query, model_type='tfidf')
        
        if not processed_query.strip():
            return []
        
        try:
            query_vector = self.tfidf_model.transform([processed_query])
        except Exception as e:
            print(f"Lỗi transform query: {e}")
            return []
            
        similarities = skpair.cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score > similarity_threshold:
                results.append({
                    'title': self.df.iloc[idx]['title'],
                    'genre': self.df.iloc[idx]['genre'],
                    'description': self.df.iloc[idx].get('description', 'N/A'),
                    'year': self.df.iloc[idx]['year'],    # <--- THÊM NĂM
                    'poster': self.df.iloc[idx]['poster'], # <--- THÊM POSTER
                    'similarity_score': float(similarity_score),
                    'original_index': int(idx)
                })
        
        return results
        
    
    def get_similar_movies(self, movie_title, top_k=10, exclude_self=True):
    # ... (Phần kiểm tra if movie_title not in self.movie_indices_map giữ nguyên)
    
    # 1. Tra cứu Index (LƯU KẾT QUẢ CỦA TRA CỨU)
        movie_idx_raw = self.movie_indices_map[movie_title]
    
    # 2. ✅ FIX LỖI: Kiểm tra nếu kết quả là Series (tức là có nhiều khớp)
        if isinstance(movie_idx_raw, pd.Series):
        # Nếu là Series, lấy chỉ mục của phần tử đầu tiên
            movie_idx = movie_idx_raw.iloc[0] 
        else:
        # Nếu là scalar (trường hợp bình thường)
            movie_idx = movie_idx_raw

    # 3. ÉP KIỂU VỀ SCALAR INT (Buộc giá trị đã trích xuất thành số nguyên)
        try:
            movie_idx = int(movie_idx)
        except ValueError:
            print(f"⚠️ Lỗi ép kiểu: Giá trị chỉ mục không phải là số nguyên.")
            return []
        # 2. Lấy vector của phim gốc và tính similarity 1×N DYNAMIC
        movie_vector = self.tfidf_matrix[movie_idx]
        similarities = skpair.cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 3. Loại bỏ phim gốc (Sử dụng boolean indexing an toàn)
        if exclude_self:
            # Sửa lỗi: Chỉ so sánh NumPy array với scalar, loại bỏ phim gốc
            sorted_indices = sorted_indices[sorted_indices != movie_idx] 
        
        # 4. Lấy top K
        top_indices = sorted_indices[:top_k]
        
        # ... (Tạo kết quả giữ nguyên) ...
        similar_movies = []
        for idx in top_indices:
            similar_movies.append({
                'title': self.df.iloc[idx]['title'],
                'genre': self.df.iloc[idx]['genre'],
                'description': self.df.iloc[idx].get('description', 'N/A'),
                'year': self.df.iloc[idx]['year'],    # <--- THÊM NĂM
                'poster': self.df.iloc[idx]['poster'], # <--- THÊM POSTER URL
                'similarity_score': float(similarities[idx]),
                'original_index': int(idx)
            })
        
        return similar_movies


    def get_similar_movies_by_index(self, movie_idx, top_k=10, exclude_self=True):
        """
        Gợi ý phim tương tự dựa trên index của phim.
        (ĐÃ FIX LỖI ÉP KIỂU SCALAR)
        """
        # Nếu movie_idx đến từ bên ngoài, ta vẫn cần đảm bảo nó là int
        movie_idx = int(movie_idx) 
        
        if movie_idx >= len(self.df) or movie_idx < 0:
            return []
            
        # ✅ Lấy vector của phim gốc và tính similarity 1×N DYNAMIC
        movie_vector = self.tfidf_matrix[movie_idx]
        similarities = skpair.cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        if exclude_self:
            # FIX LỖI: Đảm bảo phép so sánh NumPy an toàn
            sorted_indices = sorted_indices[sorted_indices != movie_idx]
        
        top_indices = sorted_indices[:top_k]
        
        # Tạo kết quả
        similar_movies = []
        for idx in top_indices:
            similar_movies.append({
                'title': self.df.iloc[idx]['title'],
                'genre': self.df.iloc[idx]['genre'],
                'description': self.df.iloc[idx].get('description', 'N/A'),
                'year': self.df.iloc[idx]['year'],    # <--- THÊM NĂM
                'poster': self.df.iloc[idx]['poster'], # <--- THÊM POSTER URL
                'similarity_score': float(similarities[idx]),
                'original_index': int(idx)
            })
            
        return similar_movies
    
    # [Các hàm khác như hybrid_search, get_popular_movies_by_genre sẽ được cập nhật tương tự]


# Hàm tiện ích để load model (Dùng trong app.py)
def load_tfidf_models(tfidf_path, matrix_path, data_path):
    # ... (Logic load giữ nguyên) ...
    try:
        tfidf_model = joblib.load(tfidf_path)
        
        import scipy.sparse as sp
        if matrix_path.endswith('.npz'):
            tfidf_matrix = sp.load_npz(matrix_path)
        else:
            tfidf_matrix = joblib.load(matrix_path)
        
        df = pd.read_csv(data_path)
        
        print("✅ Đã load TF-IDF models thành công (Memory Safe)")
        return TFIDFRecommender(tfidf_model, tfidf_matrix, df)
        
    except Exception as e:
        print(f"❌ Lỗi khi load TF-IDF models: {e}")
        return None
