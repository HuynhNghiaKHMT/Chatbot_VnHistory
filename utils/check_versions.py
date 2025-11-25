import pkg_resources

# Danh sách các gói bạn đang sử dụng trong app.py
PACKAGES_TO_CHECK = [
    "weaviate-client", 
    "sentence-transformers", 
    "python-dotenv", 
    "langchain",
    "langchain-core", 
    "langchain-community", # Cần thiết cho các thành phần cũ
    "langchain-google-genai", 
    "langchain-classic", # Chứa RetrievalQA mà bạn đang dùng
    "numpy", # Dependency của SentenceTransformer
    "scikit-learn", # Dependency của SentenceTransformer
    "pyvi", # Nếu bạn dùng pyvi trong tiền xử lý văn bản,
    "joblib",
    "streamlit"

]


def check_versions(packages):
    """Kiểm tra và in ra phiên bản của các gói đã cài đặt."""
    print("--- 🔬 KIỂM TRA PHIÊN BẢN THƯ VIỆN RAG ---")
    print("------------------------------------------")
    
    for package_name in packages:
        try:
            version = pkg_resources.get_distribution(package_name).version
            print(f"✅ {package_name:<25}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package_name:<25}: Gói chưa được cài đặt (hoặc tên không đúng).")
            
    print("------------------------------------------")

if __name__ == "__main__":
    check_versions(PACKAGES_TO_CHECK)