import re
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =======================================================
#  1. HÀM CHUẨN HOÁ TEXT CƠ BẢN (Core Normalization)
# =======================================================
def normalize_text(text: str) -> str:
    """
    Chuẩn hoá text: lowercase, remove accents (Unicode), remove special chars.
    Đây là bước cần thiết cho cả TF-IDF và SBERT.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove accents (Xóa dấu tiếng Việt, Tiếng Pháp, v.v. - Dùng unicodedata)
    # Rất quan trọng để đảm bảo tính nhất quán của từ khóa
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    
    # 3. Remove special chars (Chỉ giữ lại chữ cái và số)
    # Dùng regex để thay thế mọi thứ không phải chữ cái, số, hoặc khoảng trắng bằng khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text)
    
    # 4. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# =======================================================
# 2. HÀM LOẠI STOPWORDS (Dành riêng cho TF-IDF)
# =======================================================
def remove_stopwords(text: str) -> str:
    """
    Loại bỏ các từ vô nghĩa (stopwords) khỏi chuỗi văn bản.
    """
    if not isinstance(text, str):
        return ""
        
    words = text.split()
    # Sử dụng bộ stopwords chuẩn của Scikit-learn
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)


# =======================================================
#  3. HÀM HỖ TRỢ: GỘP CÁC TRƯỜNG DỮ LIỆU AN TOÀN (CÓ TRỌNG SỐ)
# =======================================================
def _build_text_core(row, weighted_fields: list) -> str:
    """
    Gộp các trường dữ liệu được chỉ định thành một chuỗi duy nhất,
    áp dụng trọng số bằng cách nhân bản nội dung.

    weighted_fields: List các tuple (field_name, weight_factor).
    Ví dụ: [('genre', 4), ('director', 3)]
    """
    parts = []
    
    # Lặp qua danh sách các tuple (tên trường, trọng số)
    for field, weight in weighted_fields:
        value = row.get(field)
        
        if pd.notna(value) and str(value).strip():
            content = str(value)
            
            # ÁP DỤNG TRỌNG SỐ: Nhân bản chuỗi
            weighted_content = (content + ' ') * weight
            
            parts.append(weighted_content.strip())
            
    # Nối các phần lại với nhau bằng dấu chấm hoặc khoảng trắng
    return ". ".join(parts)


# =======================================================
# 4. TẠO TEXT ĐẦU VÀO CHO TF-IDF (similarity_text)
# =======================================================
def build_similarity_text(row):
    """
    Tạo text cho TF-IDF: Gộp các trường, chuẩn hóa, và loại bỏ stopwords,
    ÁP DỤNG TRỌNG SỐ cho các trường quan trọng.
    """
    # 1. Định nghĩa Trọng số cho từng trường (W = số lần lặp lại)
    WEIGHTED_FIELDS = [
        ('title', 2),
        ('genre', 4),
        ('director', 3),
        ('cast', 3),
        ('description', 3),
        ('review', 1),
    ] 

    # 2. Gộp văn bản và áp dụng nhân bản trọng số
    joined_text = _build_text_core(row, WEIGHTED_FIELDS)

    # 3. Chuẩn hóa (Normalize)
    cleaned = normalize_text(joined_text)

    # 4. Loại bỏ Stopwords
    cleaned = remove_stopwords(cleaned)

    return cleaned
# =======================================================
# 5. TẠO TEXT ĐẦU VÀO CHO SBERT (full_text)
# =======================================================
def build_full_text(row):
    """
    Tạo text cho SBERT: Gộp tất cả các trường, chỉ chuẩn hóa (giữ lại stopwords).
    SBERT cần ngữ cảnh đầy đủ.
    """
    # # Gộp tất cả các trường (Giữ nguyên field_list như trên)
    # field_list = ['title', 'genre', 'description', 'review', 'director', 'cast']
    
    # joined_text = _build_text_core(row, field_list)

    # 1. Định nghĩa các trường với TRỌNG SỐ W=1
    SBERT_FIELDS = [
        ('title', 1),
        ('genre', 1),
        ('description', 1),
        ('review', 1),
        ('director', 1),
        ('cast', 1)
    ]
    
    # 2. Gộp văn bản và truyền vào hàm _build_text_core
    joined_text = _build_text_core(row, SBERT_FIELDS) 
    
    # Chỉ Chuẩn hóa (Normalize)
    cleaned = normalize_text(joined_text)
    
    return cleaned


# =======================================================
# 6. TIỀN XỬ LÝ CÂU TRUY VẤN (Dùng cho Semantic Search)
# =======================================================
def preprocess_query(query: str, model_type: str = 'sbert') -> str:
    """
    Tiền xử lý câu truy vấn của người dùng.
    query: Câu hỏi người dùng nhập vào.
    model_type: 'tfidf' hoặc 'sbert'.
    """
    # 1. Chuẩn hóa cơ bản
    query = normalize_text(query)

    if model_type == 'tfidf':
        # 2. Loại bỏ stopwords (Quan trọng để khớp từ khóa với ma trận TF-IDF đã làm sạch)
        query = remove_stopwords(query)
    
    # Đối với SBERT: chỉ cần normalize_text (đã thực hiện ở bước 1)
    
    return query


# =======================================================
# 7. HÀM CHÍNH: Load cleaned dataset + tạo các cột xử lý
# =======================================================
def load_and_process_data(path: str = "../data/clean_movies.csv") -> pd.DataFrame:
    """
    Load dữ liệu sạch, tạo các cột text đã được tiền xử lý cho 2 mô hình.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn {path}. Hãy kiểm tra lại.")
        return pd.DataFrame()
    
    print("--- Bắt đầu tạo cột văn bản đầu vào cho mô hình ---")
    
    # 1. Tạo cột cho TF-IDF (similarity_text)
    df["similarity_text"] = df.apply(build_similarity_text, axis=1)

    # 2. Tạo cột cho SBERT (full_text)
    df["full_text"] = df.apply(build_full_text, axis=1)

    print("--- Hoàn tất tạo văn bản xử lý ---")
    return df