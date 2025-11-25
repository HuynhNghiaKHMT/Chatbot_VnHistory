import os
import weaviate
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- LangChain/Weaviate Imports ---
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from typing import List, Dict, Any, Optional

# --- Import Predictor ---
# Đảm bảo đường dẫn và tên biến export trong predict.py là đúng
import sys
sys.path.append(os.path.join(os.getcwd(), 'document_classification'))
try:
    from document_classification.predict import DocumentPredictor 
except ImportError:
    st.error("Lỗi Import: Vui lòng đảm bảo file 'document_classification/predict.py' tồn tại.")
    DocumentPredictor = None

# ----------------------------------------------------------------------
# 1. HÀM VÀ LỚP LOGIC RAG
# ----------------------------------------------------------------------

def reciprocal_rank_fusion(results: List[Any], k: int = 60) -> List[Dict[str, Any]]:
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
        
        where_filter: Optional[Filter] = None
        predicted_period: str = "Không lọc"
        
        if self.use_filter and DocumentPredictor:
            predicted_period = DocumentPredictor.predict(query)
            where_filter = Filter.by_property("period").equal(predicted_period)
            
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
                "rrf_score": item["score"],
                "predicted_period": predicted_period 
            }
            documents.append(
                Document(page_content=obj.properties.get("context", ""), metadata=metadata)
            )
        
        return documents

# ----------------------------------------------------------------------
# 2. KHỞI TẠO VÀ CẤU HÌNH STREAMLIT
# ----------------------------------------------------------------------

load_dotenv() 

def setup_rag_system(temperature: float, k_value: int, search_mode: str, use_filter: bool):
    """Thực hiện khởi tạo các thành phần RAG."""
    
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    COLLECTION_NAME = "History"
    LOCAL_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_NAME")

    embed_model = SentenceTransformer(LOCAL_MODEL_PATH)
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
    
    if not client.is_connected():
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
    
    return qa_chain, client

def initialize_rag_system(temp, k, mode, filter):
    """Thực hiện khởi tạo hệ thống RAG và lưu vào Session State."""
    try:
        with st.spinner("⏳ Đang Khởi tạo Hệ thống RAG..."):
            qa_chain, client = setup_rag_system(temp, k, mode, filter)
            
            st.session_state['qa_chain'] = qa_chain
            st.session_state['weaviate_client'] = client
            st.session_state['rag_initialized'] = True
            st.session_state['last_config'] = (temp, k, mode, filter)
        
        st.success("✅ Hệ thống RAG đã được khởi tạo thành công! Bạn có thể đặt câu hỏi.")
        st.session_state.current_config_status = "Đã khởi tạo"
        
    except Exception as e:
        st.error(f"❌ Lỗi Khởi tạo: {e}")
        st.session_state['rag_initialized'] = False
        st.session_state.current_config_status = "Lỗi khởi tạo"

# ----------------------------------------------------------------------
# 3. GIAO DIỆN STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(page_title="🇻🇳 Chatbot Lịch sử Việt Nam (RAG)", layout="wide")

# Yêu cầu 2: Thay đổi tiêu đề chính
st.title("Chatbot Lịch sử Việt Nam (RAG)")
st.caption("Sử dụng Gemini 2.5 Flash, Hybrid Search (RRF) & Phân loại Tài liệu")

# --- KHỞI TẠO TRẠNG THÁI CUỘC TRÒ CHUYỆN & CẤU HÌNH ---
if 'rag_initialized' not in st.session_state:
    st.session_state['rag_initialized'] = False
    st.session_state.messages = []
    st.session_state.current_config_status = "Chưa khởi tạo"
    st.session_state.last_config = (0.1, 4, 'hybrid', True) # Giá trị mặc định

# --- Cột bên (Sidebar) ---
with st.sidebar:
    # Yêu cầu 1: Thay đổi tiêu đề sidebar
    st.header("ViHistory Chatbot")

    temperature = st.slider("1️⃣ Độ sáng tạo (Temperature)", min_value=0.0, max_value=1.0, value=st.session_state.last_config[0], step=0.05, key="temp_slider")
    k_value = st.slider("2️⃣ Số lượng Context (K)", min_value=1, max_value=10, value=st.session_state.last_config[1], step=1, key="k_slider")
    search_mode = st.radio(
        "3️⃣ Phương pháp Truy xuất", 
        ('hybrid', 'semantic', 'keyword'),
        format_func=lambda x: {'semantic': 'Ngữ nghĩa (Vector)', 'keyword': 'Từ khóa (BM25)', 'hybrid': 'Kết hợp (RRF)'}[x],
        index=['hybrid', 'semantic', 'keyword'].index(st.session_state.last_config[2]),
        key="search_mode_radio"
    )
    # Yêu cầu 3: Thay đổi tên checkbox
    use_filter = st.checkbox("4️⃣ Bộ lọc câu hỏi", value=st.session_state.last_config[3], key="filter_checkbox")
    
    st.markdown("---")
    
    # Yêu cầu 4: Hiển thị trạng thái cấu hình
    st.metric("Trạng thái Hệ thống", st.session_state.current_config_status)
    st.caption("Vui lòng khởi tạo lại hệ thống khi thay đổi các tùy chọn!") 

    if st.button("⚙️ Khởi tạo Hệ thống RAG"):
        initialize_rag_system(temperature, k_value, search_mode, use_filter)


# --- Khu vực Chat Chính ---

# 1. Hiển thị Lịch sử Chat
for message in st.session_state.messages:
    # Yêu cầu 5: Đảm bảo nguồn xuất hiện ngay lập tức
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Nếu là Assistant, hiển thị nguồn ngay bên dưới câu trả lời
        if message["role"] == "assistant" and "sources" in message:
            
            # Hiển thị thông tin Lọc (nếu có)
            if message['filter_info']:
                st.info(message['filter_info'])
                
            # Mở expander cho Nguồn Context
            # with st.expander(f"📚 Xem {len(message['sources'])} Đoạn Nguồn Context"):
            #     for i, doc in enumerate(message['sources']):
            #         metadata = doc.metadata
                    
            #         # Yêu cầu 6: Loại bỏ Nguồn UUID và RRF Score
            #         st.markdown(f"**Tài liệu {i+1}: {metadata.get('period', 'N/A')}**")
            #         st.code(doc.page_content, language='markdown')


# 2. Nhập Câu hỏi mới (Chat Input)
if st.session_state['rag_initialized']:
    
    user_query = st.chat_input("Nhập câu hỏi của bạn về Lịch sử Việt Nam...")

    if user_query:
        # A. Thêm câu hỏi người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Hiển thị ngay lập tức để tạo hiệu ứng chat
        with st.chat_message("user"):
            st.markdown(user_query)

        # B. Lấy câu trả lời và hiển thị
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang xử lý..."):
                try:
                    qa_chain = st.session_state['qa_chain']
                    result = qa_chain.invoke({"query": user_query}) 

                    response = result["result"]
                    source_documents = result["source_documents"]
                    
                    # Chuẩn bị thông tin lọc để hiển thị trong message (Yêu cầu 2: Không hiển thị khi Filter TẮT)
                    filter_info = None
                    if source_documents and qa_chain.retriever.use_filter:
                        predicted = source_documents[0].metadata.get('predicted_period', 'N/A')
                        filter_info = f"🔥 **Đã Lọc theo Chủ đề:** **{predicted}** (Dự đoán từ câu hỏi)"

                    # Hiển thị câu trả lời (sau khi xử lý)
                    st.markdown(response)

                    if source_documents:
                        # Logic hiển thị filter_info và expander cho sources
                        if filter_info:
                            st.info(filter_info)
                        
                        with st.expander(f"📚 Xem {len(source_documents)} Đoạn Nguồn Context"):
                            for i, doc in enumerate(source_documents):
                                metadata = doc.metadata
                                st.markdown(f"**Tài liệu {i+1}: {metadata.get('period', 'N/A')}**")
                                st.code(doc.page_content, language='markdown')
                    
                    # C. Thêm câu trả lời và nguồn vào lịch sử
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": source_documents,
                        "filter_info": filter_info
                    })
                    
                except Exception as e:
                    error_message = f"❌ Đã xảy ra lỗi trong quá trình RAG: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    # Hiển thị thông báo khi chưa khởi tạo
    st.info("Vui lòng cấu hình các tùy chọn và nhấn **'⚙️ Khởi tạo Hệ thống RAG'** ở sidebar để bắt đầu.")