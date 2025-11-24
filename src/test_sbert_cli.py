# test_load_sbert.py
"""
🧪 Test Load SBERT Model
Kiểm tra việc load SBERT model và embeddings
"""

import sys
import os
import pandas as pd

# Thêm path để import từ src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_load_sbert():
    """Test load SBERT model"""
    print("🧪 BẮT ĐẦU TEST LOAD SBERT MODEL")
    
    try:
        from src.recommender_sbert import load_sbert_models
        
        # Các đường dẫn cần kiểm tra
        paths = {
            'SBERT_MODEL': "../models/sbert_model",
            'EMBEDDINGS': "../models/sbert_embeddings.pt", 
            'DATA': "../data/clean_movies.csv"
        }
        
        print("\n🔍 KIỂM TRA FILE TỒN TẠI:")
        for name, path in paths.items():
            exists = os.path.exists(path)
            status = "✅ TỒN TẠI" if exists else "❌ KHÔNG TỒN TẠI"
            print(f"   {name}: {path} - {status}")
            
            if not exists:
                print(f"      📁 Kiểm tra thư mục: {os.listdir('../models/')}")
        
        print("\n🔄 ĐANG LOAD MODEL...")
        recommender = load_sbert_models(
            model_path=paths['SBERT_MODEL'],
            embeddings_path=paths['EMBEDDINGS'],
            data_path=paths['DATA']
        )
        
        if recommender:
            print("✅ LOAD MODEL THÀNH CÔNG!")
            print(f"📊 Dataset: {len(recommender.df)} phim")
            print(f"📐 Embeddings shape: {recommender.sbert_embeddings.shape}")
            print(f"🔢 Embeddings type: {type(recommender.sbert_embeddings)}")
            
            # Test search cơ bản
            print("\n🧪 TEST SEARCH CƠ BẢN:")
            results = recommender.search_movies("action movie", top_k=3)
            print(f"   Kết quả tìm kiếm: {len(results)} phim")
            for i, movie in enumerate(results):
                print(f"     {i+1}. {movie['title']} - {movie['similarity_score']:.4f}")
                
        else:
            print("❌ LOAD MODEL THẤT BẠI")
            
    except Exception as e:
        print(f"💥 LỖI: {e}")
        import traceback
        traceback.print_exc()

def check_model_files():
    """Kiểm tra chi tiết các file trong thư mục models"""
    print("\n📁 KIỂM TRA CHI TIẾT THƯ MỤC MODELS:")
    
    models_dir = "../models"
    if os.path.exists(models_dir):
        items = os.listdir(models_dir)
        print(f"   Các file trong {models_dir}:")
        for item in items:
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/ (thư mục)")
                # Liệt kê file trong thư mục con
                sub_items = os.listdir(item_path)
                for sub_item in sub_items[:5]:  # Hiển thị 5 file đầu
                    print(f"      📄 {sub_item}")
            else:
                print(f"   📄 {item}")
    else:
        print(f"   ❌ Thư mục {models_dir} không tồn tại")

if __name__ == "__main__":
    check_model_files()
    test_load_sbert()