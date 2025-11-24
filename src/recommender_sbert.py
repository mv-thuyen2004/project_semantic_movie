# src/recommender_sbert.py
"""
🎬 Hệ thống Gợi ý Phim dựa trên SBERT (Sentence-BERT)
Semantic Search và Content-Based Recommendation sử dụng embeddings

LƯU Ý: File này là THƯ VIỆN, chỉ để import vào app.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import sys
import os

# Thêm path để import từ src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_query

class SBERTRecommender:
    """
    Lớp gợi ý phim sử dụng SBERT embeddings (Memory Safe - Dynamic Calculation)
    """
    
    def __init__(self, sbert_model, sbert_embeddings, df):
        """
        Khởi tạo recommender với SBERT model và embeddings
        
        Args:
            sbert_model: SBERT model đã loaded
            sbert_embeddings: Embeddings của toàn bộ dataset (tensor hoặc numpy array)
            df: DataFrame chứa thông tin phim
        """
        self.sbert_model = sbert_model
        self.sbert_embeddings = sbert_embeddings
        self.df = df
        
        # Tạo mapping để tra cứu nhanh title -> index
        self.movie_indices_map = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        # Chuyển embeddings sang numpy nếu là tensor
        if torch.is_tensor(self.sbert_embeddings):
            self.sbert_embeddings = self.sbert_embeddings.cpu().numpy()
        
        print("✅ SBERT Recommender khởi tạo thành công")
        print(f"📊 Embeddings shape: {self.sbert_embeddings.shape}")
    
    def search_movies(self, query, top_k=10, similarity_threshold=0.3):
        """
        Tìm kiếm phim dựa trên query text sử dụng SBERT embeddings
        
        Args:
            query (str): Câu truy vấn của người dùng
            top_k (int): Số kết quả trả về
            similarity_threshold (float): Ngưỡng similarity tối thiểu
            
        Returns:
            list: Danh sách dictionary chứa thông tin phim
        """
        # Preprocess query (dùng model_type='sbert' - giữ nguyên ngữ nghĩa)
        processed_query = preprocess_query(query, model_type='sbert')
        
        if not processed_query.strip():
            return []
        
        try:
            # Tính embedding cho query
            query_embedding = self.sbert_model.encode(
                [processed_query],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Chuyển sang numpy để tính similarity
            query_embedding_np = query_embedding.cpu().numpy()
            
        except Exception as e:
            print(f"Lỗi encode query: {e}")
            return []
        
        # Tính cosine similarity 1×N
        similarities = cosine_similarity(query_embedding_np, self.sbert_embeddings).flatten()
        
        # Lấy top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Tạo kết quả
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            
            # SBERT thường có similarity scores thấp hơn TF-IDF
            if similarity_score > similarity_threshold:
                results.append({
                    'title': self.df.iloc[idx]['title'],
                    'genre': self.df.iloc[idx]['genre'],
                    'description': self.df.iloc[idx].get('description', ''),
                    'year': self.df.iloc[idx].get('year', 'N/A'),
                    'poster': self.df.iloc[idx].get('poster', ''),
                    'similarity_score': float(similarity_score),
                    'original_index': int(idx)
                })
        
        return results
    
    def get_similar_movies(self, movie_title, top_k=10, exclude_self=True):
        """
        Gợi ý phim tương tự dựa trên SBERT embeddings
        
        Args:
            movie_title (str): Tên phim cần tìm phim tương tự
            top_k (int): Số phim tương tự trả về
            exclude_self (bool): Có loại bỏ phim gốc khỏi kết quả không
            
        Returns:
            list: Danh sách phim tương tự
        """
        # Tìm index của phim
        if movie_title not in self.movie_indices_map:
            print(f"⚠️ Không tìm thấy phim: {movie_title}")
            return []
        
        movie_idx = self.movie_indices_map[movie_title]
        
        # Lấy embedding của phim gốc và tính similarity 1×N
        movie_embedding = self.sbert_embeddings[movie_idx].reshape(1, -1)
        similarities = cosine_similarity(movie_embedding, self.sbert_embeddings).flatten()
        
        # Sắp xếp và lấy top K
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Loại bỏ phim gốc nếu cần
        if exclude_self:
            sorted_indices = sorted_indices[sorted_indices != movie_idx]
        
        top_indices = sorted_indices[:top_k]
        
        # Tạo kết quả
        similar_movies = []
        for idx in top_indices:
            similar_movies.append({
                'title': self.df.iloc[idx]['title'],
                'genre': self.df.iloc[idx]['genre'],
                'description': self.df.iloc[idx].get('description', ''),
                'year': self.df.iloc[idx].get('year', 'N/A'),
                'poster': self.df.iloc[idx].get('poster', ''),
                'similarity_score': float(similarities[idx]),
                'original_index': int(idx)
            })
        
        return similar_movies
    
    def hybrid_search(self, query, top_k=10):
        """
        Tìm kiếm lai giữa semantic search và content-based recommendation
        
        Args:
            query (str): Câu truy vấn
            top_k (int): Số kết quả trả về
            
        Returns:
            list: Danh sách kết quả kết hợp
        """
        # Tìm kiếm semantic với SBERT
        semantic_results = self.search_movies(query, top_k=top_k//2)
        
        # Nếu có kết quả, lấy phim đầu tiên và gợi ý phim tương tự
        if semantic_results:
            best_match_idx = semantic_results[0]['original_index']
            similar_movies = self.get_similar_movies_by_index(best_match_idx, top_k=top_k//2)
            
            # Kết hợp kết quả (loại bỏ trùng lặp)
            combined_results = semantic_results + similar_movies
            
            # Loại bỏ trùng lặp dựa trên title
            seen_titles = set()
            unique_results = []
            
            for movie in combined_results:
                if movie['title'] not in seen_titles:
                    seen_titles.add(movie['title'])
                    unique_results.append(movie)
            
            return unique_results[:top_k]
        else:
            return self.search_movies(query, top_k=top_k)


# Hàm tiện ích để load model - DÙNG TRONG APP.PY
def load_sbert_models(model_path, embeddings_path, data_path):
    """
    Load SBERT models và embeddings từ file
    
    Args:
        model_path (str): Đường dẫn đến SBERT model
        embeddings_path (str): Đường dẫn đến embeddings
        data_path (str): Đường dẫn đến dữ liệu phim
        
    Returns:
        SBERTRecommender: Instance của recommender
    """
    # Tên mô hình fallback
    FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_model = None
    
    try:
        # 1. KIỂM TRA VÀ XÁC ĐỊNH THIẾT BỊ
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Using device: {device}")
        
        # 2. THỬ TẢI MODEL TỪ ĐƯỜNG DẪN CỤC BỘ
        print("🔄 Đang thử load SBERT model từ local...")
        sbert_model = SentenceTransformer(model_path, device=device)
        print("✅ Load model local thành công.")
        
    except Exception as e:
        # 3. NẾU LỖI, TẢI DỰ PHÒNG TỪ HUGGING FACE
        print(f"⚠️ Lỗi tải model local: {e}")
        print(f"🔄 Đang tải dự phòng từ Hugging Face...")
        try:
            sbert_model = SentenceTransformer(FALLBACK_MODEL_NAME, device=device)
            print("✅ Tải fallback model từ HF thành công.")
        except Exception as e_hf:
            print(f"❌ Lỗi: Không thể tải fallback model từ HF: {e_hf}")
            return None

    # 4. TẢI EMBEDDINGS VÀ DATA
    try:
        print("🔄 Đang load embeddings...")
        if embeddings_path.endswith('.pt'):
            sbert_embeddings = torch.load(embeddings_path, map_location=device)
        else:
            sbert_embeddings = np.load(embeddings_path)
        
        print("🔄 Đang load data...")
        df = pd.read_csv(data_path)
        
        print("✅ Đã load SBERT models thành công")
        return SBERTRecommender(sbert_model, sbert_embeddings, df)
        
    except Exception as e_final:
        print(f"❌ Lỗi khi load embeddings/data: {e_final}")
        return None


def load_sbert_models_from_huggingface(model_name="sentence-transformers/all-MiniLM-L6-v2", embeddings_path=None, data_path=None):
    """
    Load SBERT model từ HuggingFace và embeddings từ file
    
    Args:
        model_name (str): Tên model trên HuggingFace
        embeddings_path (str): Đường dẫn đến embeddings
        data_path (str): Đường dẫn đến dữ liệu phim
        
    Returns:
        SBERTRecommender: Instance của recommender
    """
    try:
        # Xác định device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Using device: {device}")
        
        # Load SBERT model từ HuggingFace
        print(f"🔄 Đang load SBERT model: {model_name}")
        sbert_model = SentenceTransformer(model_name, device=device)
        
        # Load embeddings
        if embeddings_path and os.path.exists(embeddings_path):
            print("🔄 Đang load embeddings...")
            if embeddings_path.endswith('.pt'):
                sbert_embeddings = torch.load(embeddings_path, map_location=device)
            else:
                sbert_embeddings = np.load(embeddings_path)
        else:
            raise FileNotFoundError(f"Embeddings file không tồn tại: {embeddings_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        print("✅ Đã load SBERT models thành công")
        return SBERTRecommender(sbert_model, sbert_embeddings, df)
        
    except Exception as e:
        print(f"❌ Lỗi khi load SBERT models: {e}")
        return None


# KHÔNG CÓ PHẦN DEMO/MAIN - FILE CHỈ ĐỂ IMPORT