import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.collections import Collection
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from typing import List

# --- Thêm Import Predictor ---
from document_classification.predict import DocumentPredictor

# ----------------------------------------------------------------------
# KHỐI LOGIC HỢP NHẤT RRF VÀ CUSTOM RETRIEVER
# ----------------------------------------------------------------------

# Hàm RRF đã điều chỉnh để hoạt động với LangChain (hợp nhất các đối tượng Weaviate)
def reciprocal_rank_fusion(results, k=60):
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

# Custom Retriever để thực hiện Hybrid Search (Vector + BM25)
class HybridRetriever(BaseRetriever):
    """Retriever tùy chỉnh sử dụng Weaviate near_vector và bm25."""
    history_collection: Collection 
    embed_model: SentenceTransformer
    k: int = 4 # Số lượng tài liệu sẽ trả về cho LLM

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:

        # 0. PHÂN LOẠI TÀI LIỆU ===
        predicted_period = DocumentPredictor.predict(query)
        print(f"\n=== [DEBUG 0] CHỦ ĐỀ DỰ ĐOÁN: {predicted_period} ===")
        where_filter = Filter.by_property("period").equal(predicted_period)
        
        # 1. Tạo Vector từ mô hình cục bộ
        query_vector = self.embed_model.encode(query).tolist() 

        # 2. Vector Search (Ngữ nghĩa)
        print("\n=== [DEBUG 1] KẾT QUẢ TRUY XUẤT NGỮ NGHĨA (VECTOR SEARCH) ===")
        vector_results = self.history_collection.query.near_vector(
            near_vector=query_vector,
            limit=self.k, # Lấy nhiều kết quả hơn cho RRF
            return_properties=["context", "period"],
            return_metadata=MetadataQuery(distance=True),
            # filters=where_filter # Tên tham số là 'filters'
        )
        for i, obj in enumerate(vector_results.objects):
            context = obj.properties.get("context", "")
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance else "N/A"
            print(f"[{i+1}] (Distance: {distance:.4f}) [UUID: {obj.uuid}] - Context: {context[:100]}...")


        # 3. BM25 Search (Từ khóa)
        print("\n=== [DEBUG 2] KẾT QUẢ TRUY XUẤT TỪ KHÓA (BM25 SEARCH) ===")
        bm25_results = self.history_collection.query.bm25(
            query=query,
            limit=self.k, # Lấy nhiều kết quả hơn
            return_properties=["context", "period"],
            return_metadata=MetadataQuery(),
            # filters=where_filter # Tên tham số là 'filters'
        )
        for i, obj in enumerate(bm25_results.objects):
            context = obj.properties.get("context", "")
            print(f"[{i+1}] [UUID: {obj.uuid}] - Context: {context[:100]}...")


        # 4. Hợp nhất bằng RRF
        fused_objects = reciprocal_rank_fusion([
            vector_results.objects,
            bm25_results.objects
        ], k=60)
        
        print("\n=== [DEBUG 3] KẾT QUẢ HỢP NHẤT RRF (TOP 8) ===")
        # In ra 8 kết quả hàng đầu sau khi hợp nhất để xem RRF hoạt động
        for i, item in enumerate(fused_objects[:4]):
            obj = item["object"]
            context = obj.properties.get("context", "")
            score = item["score"]
            print(f"[{i+1}] (RRF Score: {score:.4f}) [UUID: {obj.uuid}] - Context: {context[:100]}...")

        # 5. Chuyển đổi các đối tượng Weaviate thành Document của LangChain
        # Chỉ lấy top K (4 documents)
        documents = []
        for item in fused_objects[:self.k]:
            obj = item["object"]
            # Chuyển đổi thuộc tính Weaviate thành metadata của LangChain Document
            metadata = {
                "period": obj.properties.get("period", "N/A"),
                "source_uuid": str(obj.uuid),
                "rrf_score": item["score"]
            }
            documents.append(
                Document(
                    page_content=obj.properties.get("context", ""),
                    metadata=metadata
                )
            )
        
        print(f"\n<<< TRẢ VỀ {len(documents)} DOCUMENT CHO LLM (Đã Lọc theo {predicted_period})>>>")
        return documents

# ----------------------------------------------------------------------
# ỨNG DỤNG RAG CHÍNH
# ----------------------------------------------------------------------

# 1. Tải Biến Môi trường và Khởi tạo Model
load_dotenv() 

# Lấy thông tin kết nối
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
COLLECTION_NAME = "History" 
# Sử dụng đường dẫn mô hình cục bộ của bạn
LOCAL_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_NAME")

client = None
try:
    # Khởi tạo mô hình nhúng cục bộ
    embed_model = SentenceTransformer(LOCAL_MODEL_PATH)

    # 2. Thiết lập Kết nối Weaviate
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
    
    if not client.is_ready():
        raise ConnectionError("Weaviate client is not ready.")
    
    history_collection = client.collections.get(COLLECTION_NAME)

    # 3. Khởi tạo LLM (Sử dụng Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    # 4. Khởi tạo Custom Hybrid Retriever
    retriever = HybridRetriever(
        history_collection=history_collection,
        embed_model=embed_model,
        k=4 # LLM sẽ nhận 4 context documents từ Hybrid Search
    )

    # 5. Thiết lập Prompt (Hướng dẫn LLM)
    template = """
    Bạn là một trợ lý thông minh về Lịch sử Việt Nam. 
    Hãy sử dụng các đoạn ngữ cảnh (Context) được cung cấp dưới đây để trả lời câu hỏi một cách chi tiết và trung thực. 
    
    Ngữ cảnh: {context}

    Câu hỏi: {question}

    Câu trả lời chi tiết:
    """
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(template)

    # Nếu thông tin trong ngữ cảnh không đủ hoặc không liên quan, hãy trả lời "Tôi không thể trả lời câu hỏi này dựa trên thông tin được cung cấp."

    # 6. Tạo RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_CUSTOM}
    )

    # 7. Chạy Truy vấn RAG
    HỎI = "Chiến dịch điện biên phủ xảy ra vào năm nào?"
    
    print(f"\n❓ Câu hỏi: {HỎI}")
    print("---------------------------------------")
    
    # Sử dụng .invoke()
    result = qa_chain.invoke({"query": HỎI}) 

    # 8. In kết quả
    print("\n---------------------------------------")
    print("🤖 Câu trả lời từ Gemini:")
    print(result["result"])
    print("\n📚 Nguồn Context được sử dụng (Được chọn bởi RRF):")
    for doc in result["source_documents"]:
        print(f"- [Nguồn: {doc.metadata.get('period', 'N/A')}, RRF Score: {doc.metadata.get('rrf_score', 0.0):.4f}] {doc.page_content[:100]}...")

except Exception as e:
    print(f"\n❌ Đã xảy ra lỗi trong quá trình RAG: {e}")

finally:
    if client and client.is_connected():
        client.close()