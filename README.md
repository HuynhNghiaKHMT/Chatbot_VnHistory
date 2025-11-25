# CS431 - Vietnam History Q&A System (RAG via Streamlit)
Dự án này triển khai một **Hệ thống Truy vấn Lịch sử Việt Nam**(Retrieval-Augmented Generation - RAG) giai đoạn 1945 - 1975, sử dụng Streamlit làm giao diện tương tác. Hệ thống kết hợp khả năng ngôn ngữ mạnh mẽ của mô hình Gemini AI với cơ sở tri thức lịch sử được lưu trữ trong **Vector Database Weaviate**, đồng thời tối ưu hóa truy xuất thông tin bằng các kỹ thuật như **Hybrid Search (Vector + BM25), RRF (Reciprocal Rank Fusion)** và **phân loại/lọc dựa trên câu hỏi người dùng**.

## 📦 Công nghệ và Thư viện sử dụng

- **LLM**: `Gemini 2.5 Flash` qua API của Google.
- **Vector Database**: `Weaviate` (Cloud) để lưu trữ và truy vấn ngữ cảnh lịch sử.
- **Embedding Model**: `SentenceTransformer` (mô hình finetune [BGE-M3-Viet](https://huggingface.co/AITeamVN/Vietnamese_Embedding)) để tạo vector ngữ nghĩa cho cả tài liệu và câu hỏi.
- **RAG**: Triển khai `Hybrid Search (Vector + BM25)` và `RRF (Reciprocal Rank Fusion)` để tối ưu hóa việc tìm kiếm ngữ cảnh.
- **Phân loại & Lọc**: Sử dụng mô hình `SVC` để dự đoán Thời kỳ lịch sử từ câu hỏi và lọc tài liệu trong Weaviate.
- **Giao diện Web**: `Streamlit` để cung cấp một ứng dụng Chatbot tương tác cao.

## 📂 Cấu trúc thư mục
```bash
Chatbot_VnHistory
├── .streamlit/
├── assets/
├── bge_m3_lora-embedding-models # Mô hình của bạn
├── document_classification/
    ├── models/
    ├── vncorenlp/
    ├── predict.py
    ├── preprocess.py
    └── vietnamese-stopwords.txt
├── Utils/
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md

```
## 🚀 Cài đặt và sử dụng
Để chạy dự án, hãy làm theo các bước sau:

### 1. Clone Repository

```bash
git clone https://github.com/HuynhNghiaKHMT/Chatbot_VnHistory.git
cd Chatbot_VnHistory
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
venv\Scripts\activate  # Trên Windows
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 4. Thiết lập Khóa API
```bash
# --- AI/LLM Keys ---
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# --- Weaviate Configuration ---
WEAVIATE_URL="YOUR_WEAVIATE_CLUSTER_URL"
WEAVIATE_API_KEY="YOUR_WEAVIATE_API_KEY"
COLLECTION_NAME="YOUR_WCOLLECTION"

# --- Embedding Model ---
EMBEDDING_MODEL_NAME="YOUR_EMBEDDING_MODEL"
```

### 5. Thiết lập Môi trường
```bash
python --version
py -3.10.1
```

## 🏃 Demo
### 1. Chạy Demo ByteTrack cơ bản
```bash
python Chatbot_VnHistory.py
```
Lệnh này sẽ chạy demo chat trực tiếp trên máy tính của bạn với câu hỏi mẫu được cung cấp sẵn. Bạn sẽ thấy cách hệ thống trả lời các câu hỏi dựa trên các tài liệu liên quan được truy xuất thông qua các kỹ thuật RAG đã triển khai. Hoặc bạn có thể sử dụng file `Chatbot_VnHistory.ipynb` để thử nghiệm.

### 2. Chạy Demo với ứng dụng Streamlit
```bash
python -m streamlit run app.py
```
Lệnh này sẽ chạy demo tracking trực tiếp trên Streamlit app và hỗ trợ điều chỉnh các tham số khác nhau. Mở trình duyệt và truy cập vào địa chỉ http://localhost:8501 để sử dụng ứng dụng.

Các Tính năng RAG Tùy chỉnh (Trong Sidebar)
| Tham số | Phạm vi | Mục đích |
| :--- | :--- | :--- |
| **Độ sáng tạo (Temperature)** | 0.0 - 1.0 | Điều chỉnh mức độ sáng tạo của LLM (0.0: Thực tế, 1.0: Sáng tạo hơn). |
| **Số lượng Context (K)** | 1 - 10 | Số lượng tài liệu nguồn (Chunks) được truy xuất để đưa vào LLM. |
| **Phương pháp Truy xuất** | Semantic/Keyword/Hybrid | Lựa chọn giữa tìm kiếm Vector (Ngữ nghĩa), BM25 (Từ khóa) hoặc kết hợp cả hai bằng RRF/weight. |
| **Bộ lọc câu hỏi** | On/Off | Cho phép điều chỉnh bộ lọc của mô hình phân loại để giới hạn tài liệu theo Thời kỳ lịch sử trước khi truy vấn Weaviate. |
| **Hiển thị nguồn** | On/Off | Hiển thị/Ẩn các đoạn tài liệu nguồn mà LLM đã sử dụng để tạo câu trả lời. |

## 🎞️ Video Demo
Dưới đây là một đoạn video/GIF ngắn minh họa hoạt động của ứng dụng VnHistory mà mình đã triển khai:

<!-- <img src="assets/demo.mp4" width="100%"> -->
<!-- https://github.com/user-attachments/assets/a498fc7f-1f76-4edc-b212-cb2d0e9c3cf5 -->

## 💖 Lời cảm ơn

Mình xin bày tỏ lòng biết ơn sâu sắc đến cộng đồng mã nguồn mở và các tác giả của Google (Gemini/LangChain), Weaviate và Hugging Face (Sentence Transformers). Những công cụ tiên tiến này đã tạo điều kiện cho chúng mình xây dựng một hệ thống RAG hiệu quả, phục vụ cho mục đích học tập và nghiên cứu.