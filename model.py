"""
model.py
Logic gợi ý ngành nghề dựa trên content-based filtering (embedding + cosine similarity).
Đây là phần "core" nên được test kỹ và đánh giá theo 5 trục
(Reliability - Bias - Robustness - Social Impact - Explainability)
trước khi bọc vào giao diện Streamlit.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

jobs = pd.read_csv("csv\\Career_Dataset.csv").head(20)

# add model.
_model = None


def get_model():
    """Lazy-load model để tránh load lại nhiều lần (đặc biệt quan trọng khi dùng Streamlit)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def encode_jobs():
    """Encode thành vector."""
    model = get_model()
    vectors = model.encode(jobs["Skill"].tolist())
    return vectors

def prediction_career(text: str):
    if not text or not text.strip():
        raise ValueError("Mô tả học sinh không được để trống")

    model = get_model()
    vector_jobs = encode_jobs()
    input_student = model.encode(text)

    similarities = cosine_similarity([input_student], vector_jobs)[0]

    result = []
    for i, similarity in enumerate(similarities):
        name = jobs["Career"].iloc[i]
        result.append(
            {
                "Ngành nghề": name,
                "Mức độ phù hợp": round(float(similarity) * 100, 1),
            }
        )

    result.sort(key=lambda x: -x["Mức độ phù hợp"])
    return result


  # Demo test ("notebook/log" không cần UI)
# if __name__ == "__main__":
#     Example = "Em thích sáng tạo, giải toán, tìm hiểu cách máy tính hoạt động và thích làm việc với dữ liệu. nhưng không thích làm Software engineer"
#     for r in prediction_career(Example):
#         print(r)
