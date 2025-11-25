from .preprocessor import VNPreprocessor, StopwordRemover
import joblib
import numpy as np
import os

# Đường dẫn tương đối đến thư mục chứa file jar và models (Định nghĩa 1 lần)
# VNCORENLP_PATH = "document_classification/vncorenlp" 

# print("VNCORENLP_PATH:", VNCORENLP_PATH)
# print("Files in vncorenlp:", os.listdir(VNCORENLP_PATH))
# print("Has models folder:", os.path.exists(os.path.join(VNCORENLP_PATH, "models")))

# Load mô hình
model = joblib.load("document_classification/model/best_svm.pkl")
vectorizer = joblib.load("document_classification/model/tfidf.pkl")
label_encoder = joblib.load("document_classification/model/label_encoder.pkl")

# Load stopwords
with open("document_classification/vietnamese-stopwords.txt", "r", encoding="utf8") as f:
    stopwords = f.read().splitlines()
stop = StopwordRemover(stopwords)

# Khởi tạo Preprocessor
pre = VNPreprocessor()

class PredictTopK:
    def __init__(self, model, vectorizer, label_encoder, k=1):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.k = k

    def preprocess(self, input_text):
        text = input_text.lower()
        text = pre.preprocess(text)
        text = stop.remove(text)
        return text

    def predict(self, input_text):
        processed = self.preprocess(input_text)
        encoded = self.vectorizer.transform([processed])
        proba = self.model.predict_proba(encoded)[0]

        top_k_idx = np.argsort(proba)[-self.k:][::-1]
        labels = self.label_encoder.inverse_transform(top_k_idx)
        # probs = proba[top_k_idx] 

        # return {
        #     "labels": labels.tolist(),
        #     "probs": [round(float(p) * 100, 2) for p in probs]
        # }
        return labels[0]

# tạo predictor
DocumentPredictor = PredictTopK(model, vectorizer, label_encoder, k=1)

# câu hỏi test
# input_text = input("input:")

# result = predictor.predict(input_text)

# print("Kết quả dự đoán:")
# print("Nhãn:", result["labels"])
# print("Xác suất:", result["probs"])
