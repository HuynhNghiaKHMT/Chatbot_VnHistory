from pyvi import ViTokenizer

class VNPreprocessor:
    def __init__(self, *args, **kwargs):
        # Không cần mô hình NLP được truyền vào nữa
        pass

    def preprocess(self, input_text):
        # Sử dụng ViTokenizer.tokenize() để tách từ
        # Kết quả là chuỗi đã được tách từ, ví dụ: "ý_nghĩa cuộc_kháng_chiến chống_Mỹ"
        return ViTokenizer.tokenize(input_text)

class StopwordRemover:
    def __init__(self, stopwords):
        self.stopwords = set(stopwords)

    def remove(self, text):
        words = text.split()
        return " ".join(w for w in words if w not in self.stopwords)