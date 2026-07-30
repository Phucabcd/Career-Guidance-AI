"""
model.py.
Đây là phần "core" nên được test kỹ và đánh giá theo 5 trục
(Reliability - Bias - Robustness - Social Impact - Explainability)
trước khi bọc vào giao diện Streamlit.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

jobs = pd.read_csv("csv/career_path_in_all_field.csv").head(30)
# Missing value 
# jobs = jobs.dropna(subset=[""])
# jobs = jobs.drop_duplicates(subset=[""])

# add model.
_model = None


def get_model():
    """Lazy-load model để tránh load lại nhiều lần (đặc biệt quan trọng khi dùng Streamlit)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["GPA", "Internships", "Projects"])
    ]
)

def encode_jobs():
    """Encode thành vector."""
    model = get_model()
    text_vectors = model.encode(jobs["Field"].tolist())
    num_vectors = preprocessor.fit_transform(jobs[["GPA", "Internships", "Projects"]])
    
    comined = np.hstack([text_vectors, num_vectors])
    return comined


def prediction_career(text: str):
    if not text or not text.strip():
        raise ValueError("Mô tả học sinh không được để trống")
    

    model = get_model()
    vector_jobs = encode_jobs()
    input_text_vec = model.encode(text)
    input_num_vec = preprocessor.transform([[extracted["gpa"], extracted["internships"], extracted["projects"]]])[0]
    input_combined = np.hstack([input_text_vec, input_num_vec])

    similarities = cosine_similarity([input_combined], vector_jobs)[0]

    result = []
    for i, similarity in enumerate(similarities):
        name = jobs["Career"].iloc[i]
        result.append(
            {
                "career_name": name,
                "match": round(float(similarity) * 100, 1),
            }
        )

    result.sort(key=lambda x: -x["match"])
    return result


# Demo test ("notebook/log" không cần UI)
if __name__ == "__main__":
    Example = "Em thích sáng tạo, giải toán, tìm hiểu cách máy tính hoạt động và thích làm việc với dữ liệu. điểm trên trường GPA 3.2 đã có kinh nghiêm interships và 3 projects cá nhân"
    for r in prediction_career(Example):
        print(r)
