import os
import streamlit as st
import time
import uuid
from datetime import datetime
from configparser import ConfigParser
from typing import List, Dict, Any, Optional

# --- CORE LOGIC IMPORTS (Weaviate/LangChain/Embedding) ---
import weaviate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# --- Import Predictor (Giữ nguyên logic của bạn) ---
import sys
# Thêm thư mục document_classification vào path
sys.path.append(os.path.join(os.getcwd(), 'document_classification')) 
try:
    from document_classification.predict import DocumentPredictor 
except ImportError:
    # print("WARNING: Lỗi Import DocumentPredictor. Chức năng lọc sẽ không hoạt động.")
    DocumentPredictor = None

# --- LOAD ENV VÀ CONFIG ---
load_dotenv() 
config = ConfigParser()
try:
    config.read(f"{os.path.dirname(os.path.abspath(__file__))}/env.ini")
except Exception:
    pass 

# ----------------------------------------------------------------------
# 1. CORE RAG LOGIC (ConfigurableHybridRetriever và RRF)
# ----------------------------------------------------------------------

def reciprocal_rank_fusion(results: List[Any], k: int = 60) -> List[Dict[str, Any]]:
    # Hàm RRF của bạn
    fused_scores = {}
    for result_list in results:
        for rank, obj in enumerate(result_list):
            uuid = str(obj.uuid)
            score = 1.0 / (k + rank + 1)
            if uuid not in fused_scores:
                fused_scores[uuid] = {"score": 0.0, "object": obj}
            fused_scores[uuid]["score"] += score
    sorted_fused_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    return sorted_fused_results

class ConfigurableHybridRetriever(BaseRetriever):
    history_collection: Collection 
    embed_model: SentenceTransformer
    k: int = 4
    search_mode: str
    use_filter: bool

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Logic truy xuất Weaviate của bạn
        where_filter: Optional[Filter] = None
        predicted_period: str = "Không lọc"
        
        if self.use_filter and DocumentPredictor:
              try:
                predicted_period = DocumentPredictor.predict(query)
                where_filter = Filter.by_property("period").equal(predicted_period)
              except Exception as e:
                print(f"Lỗi khi dự đoán/lọc: {e}")
                predicted_period = "Lỗi dự đoán"
        
        query_vector = self.embed_model.encode(query).tolist() 
        vector_results_objects = []
        bm25_results_objects = []

        if self.search_mode in ['semantic', 'hybrid']:
            vector_results = self.history_collection.query.near_vector(
                near_vector=query_vector,
                limit=self.k * 3 if self.search_mode == 'hybrid' else self.k,
                return_properties=["context", "period"],
                filters=where_filter,
            )
            vector_results_objects = vector_results.objects
            
        if self.search_mode in ['keyword', 'hybrid']:
            bm25_results = self.history_collection.query.bm25(
                query=query,
                limit=self.k * 3 if self.search_mode == 'hybrid' else self.k,
                return_properties=["context", "period"],
                filters=where_filter,
            )
            bm25_results_objects = bm25_results.objects

        final_results = []
        if self.search_mode == 'hybrid':
            fused_objects = reciprocal_rank_fusion([vector_results_objects, bm25_results_objects], k=60)
            final_results = fused_objects[:self.k]
        elif self.search_mode == 'semantic':
            final_results = [{"score": 1.0, "object": obj} for obj in vector_results_objects[:self.k]]
        elif self.search_mode == 'keyword':
            final_results = [{"score": 1.0, "object": obj} for obj in bm25_results_objects[:self.k]]

        documents = []
        for item in final_results:
            obj = item["object"]
            metadata = {
                "period": obj.properties.get("period", "N/A"),
                "source_uuid": str(obj.uuid),
                "predicted_period": predicted_period 
            }
            documents.append(
                Document(page_content=obj.properties.get("context", ""), metadata=metadata)
            )
        
        return documents
    
    # Phương thức mới để gọi RAG Chain
    def ask(self, query: str) -> tuple[str, List[Document], Optional[str]]:
        qa_chain = st.session_state['qa_chain']
        result = qa_chain.invoke({"query": query})
        
        response = result["result"]
        source_documents = result["source_documents"]
        
        # Chuẩn bị thông tin lọc
        filter_info = None
        if source_documents and self.use_filter:
            predicted = source_documents[0].metadata.get('predicted_period', 'N/A')
            filter_info = f"🔥 **Đã Lọc theo Chủ đề:** **{predicted}** (Dự đoán từ câu hỏi)"
            
        return response, source_documents, filter_info


def setup_rag_system(temperature: float, k_value: int, search_mode: str, use_filter: bool):
    """Khởi tạo các thành phần RAG và trả về qa_chain."""
    
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    COLLECTION_NAME = "History"
    LOCAL_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_NAME")

    if not all([WEAVIATE_URL, WEAVIATE_API_KEY, GEMINI_API_KEY, LOCAL_MODEL_PATH]):
        raise EnvironmentError("Thiếu biến môi trường (WEAVIATE_URL, WEAVIATE_API_KEY, GEMINI_API_KEY, EMBEDDING_MODEL_NAME)!")

    # Chắc chắn rằng mô hình Embedding đã được tải về (hoặc có thể dùng try/except)
    embed_model = SentenceTransformer(LOCAL_MODEL_PATH)
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
    history_collection = client.collections.get(COLLECTION_NAME)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )

    retriever = ConfigurableHybridRetriever(
        history_collection=history_collection,
        embed_model=embed_model,
        k=k_value,
        search_mode=search_mode,
        use_filter=use_filter,
    )

    template = """Bạn là một trợ lý thông minh về Lịch sử Việt Nam. 
    Hãy sử dụng các đoạn ngữ cảnh (Context) được cung cấp dưới đây để trả lời câu hỏi một cách chi tiết và trung thực. 
    Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời là 'Tôi không tìm thấy thông tin này trong nguồn cấp.'
    
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Câu trả lời chi tiết:
    """
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_CUSTOM}
    )
    
    return qa_chain, retriever, client

@st.cache_resource
def initialize_rag_system(temp, k, mode, filter, show_sources): # THÊM tham số show_sources
    """Thực hiện khởi tạo hệ thống RAG và lưu vào Session State."""
    try:
        with st.spinner("⏳ Đang Khởi tạo Hệ thống RAG..."):
            qa_chain, retriever, client = setup_rag_system(temp, k, mode, filter)
            
            st.session_state['qa_chain'] = qa_chain
            st.session_state['rag_retriever'] = retriever
            st.session_state['weaviate_client'] = client
            st.session_state['rag_initialized'] = True
            # LƯU 5 GIÁ TRỊ VÀO last_config
            st.session_state['last_config'] = (temp, k, mode, filter, show_sources) 
            st.session_state.current_config_status = "Đã khởi tạo"
        
        st.success("✅ Hệ thống RAG đã được khởi tạo thành công!")
        return True
    
    except Exception as e:
        st.session_state['rag_initialized'] = False
        st.session_state.current_config_status = "Lỗi khởi tạo"
        st.error(f"❌ Lỗi Khởi tạo: {e}")
        return False
    
# ----------------------------------------------------------------------
# 2. FRONTEND LOGIC & CẤU HÌNH BAN ĐẦU
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="3NHistory | Vietnam History AI",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- KHỞI TẠO TRẠNG THÁI CUỘC TRÒ CHUYỆN & CẤU HÌNH ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True 
if 'k_value' not in st.session_state:
    st.session_state.k_value = 5 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'rag_initialized' not in st.session_state:
    st.session_state['rag_initialized'] = False
    st.session_state.current_config_status = "Chưa khởi tạo"
    st.session_state.last_config = (0.1, 4, 'hybrid', True, True) # Thêm show_sources vào config
    st.session_state['rag_retriever'] = None
    st.session_state['qa_chain'] = None
# BIẾN TRẠNG THÁI MỚI ĐỂ KÍCH HOẠT RAG SAU KHI RERUN (Giải pháp cho việc hiển thị ngay lập tức)
if 'process_rag' not in st.session_state:
    st.session_state.process_rag = False

if 'temp_slider' not in st.session_state:
    st.session_state.temp_slider = st.session_state.last_config[0]
    st.session_state.k_slider = st.session_state.last_config[1]
    st.session_state.search_mode_radio = st.session_state.last_config[2]
    st.session_state.filter_checkbox = st.session_state.last_config[3]
    st.session_state.show_sources_checkbox = st.session_state.last_config[4]


# --- CSS - Dark Mode Modern Design (Đã thêm CSS khóa Sidebar) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    }
    
    /* Top Bar */
    .top-bar {
        position: fixed; top: 0; left: 0; right: 0; height: 70px; display: flex; 
        align-items: center; justify-content: space-between; padding: 0 2rem;
        z-index: 1000; background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px); border-bottom: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    .top-bar-left { display: flex; align-items: center; gap: 1rem; }
    .stats-badge {
        background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px; padding: 0.5rem 1rem; color: #c4b5fd;
        font-size: 0.875rem; font-weight: 600;
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 900px; margin: 90px auto 120px auto; padding: 0 1.5rem;
    }
    
    /* Welcome Screen */
    .welcome-container {
        max-width: 1000px; margin: 100px auto 0 auto; padding: 2rem 1.5rem;
        text-align: center; animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    
    .welcome-title {
        font-size: 3.5rem; font-weight: 900;
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #f59e0b 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 1rem; animation: glow 3s ease-in-out infinite;
    }
    @keyframes glow { 0%, 100% { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.4)); } 50% { filter: drop-shadow(0 0 40px rgba(236, 72, 153, 0.6)); } }
    
    .suggestion-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem; margin-top: 3rem;
    }
    
    /* Source Card */
    .source-card {
        background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
        transition: all 0.3s ease; backdrop-filter: blur(10px);
    }
    
    /* Sidebar (Đã thêm CSS khóa Sidebar) */
    /* Ẩn nút Hamburger menu/Collapse để ngăn đóng sidebar */
    [data-testid="stSidebarToggleButton"] {
        visibility: hidden;
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
        backdrop-filter: blur(20px);
        
        /* Khóa kích thước sidebar */
        width: 300px !important; 
        min-width: 300px !important;
        max-width: none !important; 
        transform: none !important;
    }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    section[data-testid="stSidebar"] h3 { color: #c4b5fd !important; }
    section[data-testid="stSidebar"] .stButton button {
        width: 100% !important; border-radius: 12px !important; padding: 0.75rem 1rem !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        border: none !important; color: white !important; font-weight: 600 !important; 
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 16px !important; backdrop-filter: blur(10px) !important;
        margin-bottom: 1rem !important; animation: slideIn 0.4s ease !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Giữ nguyên) ---
def save_current_chat():
    """Lưu cuộc hội thoại hiện tại"""
    if st.session_state.messages and st.session_state.current_chat_id is None:
        chat_id = str(uuid.uuid4())
        first_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), "Hội thoại mới")
        
        st.session_state.chat_history.append({
            'id': chat_id,
            'title': first_message[:50].strip() + ("..." if len(first_message) > 50 else ""),
            'messages': st.session_state.messages.copy(),
            'timestamp': datetime.now().strftime("%H:%M %d/%m")
        })

def load_chat(chat_id):
    """Load một cuộc hội thoại cũ"""
    save_current_chat()
    for chat in st.session_state.chat_history:
        if chat['id'] == chat_id:
            st.session_state.messages = chat['messages'].copy()
            st.session_state.current_chat_id = chat_id
            break
        
def delete_chat(chat_id):
    """Xóa một cuộc hội thoại"""
    st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = None

def new_chat():
    """Tạo cuộc hội thoại mới"""
    save_current_chat()
    st.session_state.messages = []
    st.session_state.current_chat_id = None


# ----------------------------------------------------------------------
# 3. GIAO DIỆN CHÍNH (SIDEBAR, TOP BAR, CHAT AREA)
# ----------------------------------------------------------------------

# --- Sidebar ---
with st.sidebar:
    # Logo và mô tả
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem; padding: 10px 0;">
            <img src="[YOUR_PUBLIC_LOGO_URL]" alt="3NHistory Logo" style="height: 45px; margin-bottom: 0.5rem;"/>
            <p style="font-size: 0.9rem; color: #a0aec0; margin: 0;">Vietnam History AI Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("➕ Cuộc hội thoại mới", use_container_width=True):
        new_chat()
        st.rerun()
    
    st.markdown("---")

    # A. Cấu hình RAG (Đã cập nhật thứ tự và thông báo)
    st.markdown("### ⚙️ Cấu hình RAG")
    
    temperature = st.slider("1️⃣ Độ sáng tạo (Temperature)", min_value=0.0, max_value=1.0, 
                            step=0.05, key="temp_slider") 
    
    k_value = st.slider("2️⃣ Số lượng Context (K)", min_value=1, max_value=10, 
                            step=1, key="k_slider") 
    
    search_mode = st.radio(
        "3️⃣ Phương pháp Truy xuất", 
        ('hybrid', 'semantic', 'keyword'),
        format_func=lambda x: {'semantic': 'Ngữ nghĩa', 'keyword': 'Từ khóa', 'hybrid': 'Kết hợp'}[x],
        key="search_mode_radio"
    )
    
    use_filter = st.checkbox("4️⃣ Bộ lọc câu hỏi", 
                             key="filter_checkbox", 
                             help="Sử dụng mô hình phân loại để lọc tài liệu theo thời kỳ.")
    
    st.session_state.show_sources = st.checkbox(
        "5️⃣ Hiển thị nguồn tham khảo", 
        key="show_sources_checkbox"
    )
        
    # Yêu cầu 1: Xóa gạch dưới trước nút Khởi tạo RAG
    st.markdown("---")
    
    # Yêu cầu 2: Thay đổi thông báo trạng thái
    if st.session_state.current_config_status == "Đã khởi tạo":
        status_message = "Hãy khởi tạo lại RAG nếu bạn thay đổi các tùy chọn"
    else:
        status_message = st.session_state.current_config_status
        
    st.markdown(f"**Trạng thái:** *{status_message}*")

    # Yêu cầu 2: Đặt nút Khởi tạo RAG ngay dưới thông báo
    if st.button("**🚀 Khởi tạo RAG**", use_container_width=True):
        # Cập nhật last_config bao gồm cả show_sources
        current_show_sources = st.session_state.show_sources_checkbox 
        st.session_state.last_config = (temperature, k_value, search_mode, use_filter, current_show_sources) 
        
        # TRUYỀN ĐỦ 5 THAM SỐ
        initialize_rag_system(temperature, k_value, search_mode, use_filter, current_show_sources) 
        st.rerun()

    # B. Lịch sử Chat (Giữ nguyên)
    st.markdown("---")
    if st.session_state.chat_history:
        st.markdown("### 💬 Lịch sử hội thoại")
        
        for chat in reversed(st.session_state.chat_history[-10:]):
            is_active = chat['id'] == st.session_state.current_chat_id
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"💭 {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                    load_chat(chat['id'])
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}", help="Xóa"):
                    delete_chat(chat['id'])
                    st.rerun()
            
            st.caption(f"🕐 {chat['timestamp']}")
            # st.markdown("---") # Bỏ gạch ngang vì caption đã có khoảng trống

    
    # D. Thống kê và Xóa (Giữ nguyên)
    st.markdown("---")
    st.markdown("### 📊 Thống kê")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Câu hỏi", st.session_state.total_questions)
    with col2:
        st.metric("Hội thoại", len(st.session_state.chat_history))
    
    st.markdown("---")
    
    if st.button("🗑️ Xóa Tất cả Dữ liệu", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.session_state['rag_initialized'] = False 
        st.session_state.current_config_status = "Chưa khởi tạo"
        st.rerun()
        
    st.markdown("---")
    st.info("""**3NHistory** sử dụng Gemini AI, Weaviate và mô hình Embedding cục bộ. Phục vụ mục đích giáo dục & nghiên cứu.""")


# --- Top Bar (Giữ nguyên) ---
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <img src="[YOUR_PUBLIC_LOGO_URL]" alt="3NHistory Logo" style="height: 40px;"/>
    </div>
    <div class="stats-badge">
        💬 {st.session_state.total_questions} câu hỏi
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Chat Area ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# --- A. Hiển thị Lời chào / Gợi ý ---
if len(st.session_state.messages) == 0:
    # Welcome Screen
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Chào mừng đến với 3NHistory</div>
        <div class="welcome-subtitle">
            🇻🇳 Trợ lý AI chuyên về lịch sử Việt Nam (giai đoạn từ năm 1945 - 1975)<br>
            Khám phá những trang sử vàng của dân tộc
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestion cards
    st.markdown('<div class="suggestion-grid">', unsafe_allow_html=True)
    cols = st.columns(4)
    suggestions = [
        {"icon": "⚔️", "title": "Chiến dịch quân sự", "text": "Điện Biên Phủ", "query": "Kể cho tôi nghe về chiến dịch Điện Biên Phủ"},
        {"icon": "🏛️", "title": "Sự kiện chính trị", "text": "Tuyên ngôn độc lập", "query": "Tuyên ngôn độc lập 1945 có ý nghĩa gì?"},
        {"icon": "🤝", "title": "Ngoại giao", "text": "Hội nghị Geneva", "query": "Hội nghị Geneva 1954 diễn ra như thế nào?"},
        {"icon": "👥", "title": "Nhân vật lịch sử", "text": "Vĩ nhân dân tộc", "query": "Vai trò của Hồ Chí Minh trong kháng chiến"}
    ]
    
    for i, col in enumerate(cols):
        with col:
            if st.button(
                f"{suggestions[i]['icon']}\n\n**{suggestions[i]['title']}**\n\n{suggestions[i]['text']}", 
                key=f"suggest_{i}",
                use_container_width=True
            ):
                if not st.session_state['rag_initialized']:
                    st.warning("⚠️ Vui lòng **Khởi tạo Hệ thống RAG** ở sidebar trước khi đặt câu hỏi!")
                else:
                    st.session_state.messages.append({"role": "user", "content": suggestions[i]['query']})
                    st.session_state.total_questions += 1
                    st.session_state.process_rag = True # Kích hoạt xử lý RAG
                    st.rerun() 

else:
    # --- B. Hiển thị Tin nhắn (Lần 1: Hiển thị User, Lần 2: Hiển thị Bot) ---
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            # Chỉ hiển thị nguồn nếu là tin nhắn assistant và cờ show_sources đang bật
            if (message["role"] == "assistant" and 
                st.session_state.show_sources_checkbox and 
                "sources" in message and 
                message["sources"]):
                
                # Hiển thị thông tin lọc
                if message.get('filter_info'):
                    st.info(message['filter_info'])
                
                # Hiển thị nguồn
                with st.expander("📚 Nguồn tham khảo"):
                    for i, doc in enumerate(message["sources"]):
                        period = doc.metadata.get('period', 'N/A')
                        content = doc.page_content[:200] + "..."
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 Nguồn {i+1}</strong> 
                            <span style="color: #c4b5fd;">(Thời kỳ: {period})</span><br>
                            <em style="color: #cbd5e1; line-height: 1.6;">{content}</em>
                        </div>
                        """, unsafe_allow_html=True)
                        
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 4. LOGIC XỬ LÝ RAG RIÊNG BIỆT (Đã Tách biệt)
# ----------------------------------------------------------------------

if st.session_state.process_rag and st.session_state.rag_initialized:
    # Reset cờ để không bị lặp vô hạn
    st.session_state.process_rag = False 
    
    # Lấy câu hỏi cuối cùng của người dùng
    user_prompt = st.session_state.messages[-1]["content"]

    # Khối xử lý RAG (Sẽ tự động hiển thị phía dưới tin nhắn người dùng vừa được hiển thị)
    try:
        # Sử dụng st.chat_message("assistant") để Streamlit biết đây là tin nhắn bot
        with st.chat_message("assistant", avatar="🤖"):
            
            # Sử dụng st.status để thông báo đang xử lý
            with st.status("🔍 Đang tra cứu tài liệu lịch sử...", expanded=True) as status:
                
                # Gọi RAG
                status.update(label="⌛ Đang tổng hợp và phân tích thông tin...", state="running")
                final_answer, context_docs, filter_info = st.session_state['rag_retriever'].ask(user_prompt)
                status.update(label="✅ Đã hoàn thành tra cứu", state="complete", expanded=False)
            
            # Hiển thị câu trả lời cuối cùng
            st.markdown(final_answer)
            
            # Chuẩn bị tin nhắn bot để lưu vào session state
            bot_message = {
                "role": "assistant",
                "content": final_answer,
                "sources": context_docs,
                "filter_info": filter_info
            }
            
            # Hiển thị thông tin lọc và nguồn tham khảo ngay tại đây
            if filter_info:
                st.info(filter_info) 
            
            if st.session_state.show_sources_checkbox and context_docs:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, doc in enumerate(context_docs):
                        period = doc.metadata.get('period', 'N/A')
                        content = doc.page_content[:200] + "..."
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 Nguồn {i+1}</strong> 
                            <span style="color: #c4b5fd;">(Thời kỳ: {period})</span><br>
                            <em style="color: #cbd5e1; line-height: 1.6;">{content}</em>
                        </div>
                        """, unsafe_allow_html=True)

        # Lưu tin nhắn bot vào Session State (sau khi đã hiển thị xong)
        st.session_state.messages.append(bot_message)
        
        # Gọi rerun cuối cùng để đảm bảo UI sạch sẽ (Ví dụ: xóa nội dung st.chat_input)
        st.rerun() 
        
    except Exception as e:
        error_msg = f"❌ Đã xảy ra lỗi trong quá trình RAG: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })
        st.rerun()


# ----------------------------------------------------------------------
# 5. INPUT CHAT SAU CÙNG (Chỉ dùng để thêm tin nhắn và RERUN)
# ----------------------------------------------------------------------

if prompt := st.chat_input("💭 Hỏi về lịch sử Việt Nam 1945-1975..."):
    # 1. Thêm câu hỏi người dùng vào messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_questions += 1
    
    # 2. Đặt cờ để kích hoạt RAG trong lần chạy lại (nếu đã khởi tạo)
    if st.session_state['rag_initialized']:
        st.session_state.process_rag = True
    else:
        # Nếu RAG chưa khởi tạo, thêm tin nhắn lỗi
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "❌ Vui lòng **Khởi tạo Hệ thống RAG** ở sidebar trước khi đặt câu hỏi!"
        })
        
    # 3. Yêu cầu chạy lại ngay lập tức để hiển thị tin nhắn người dùng (và bắt đầu RAG nếu cờ process_rag = True)
    st.rerun()