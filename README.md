# Career Guidance AI

Hệ thống **gợi ý hướng nghiệp** kết hợp:

1. **Gemini LLM** — đọc đoạn tự giới thiệu, trích xuất Field / điểm kỹ năng mềm / Skills / nghề ưu tiên–loại trừ, và giải thích kết quả  
2. **Random Forest** — dự đoán nghề nghiệp từ đặc trưng đã chuẩn hóa (Skills = cột CSV / Gemini, không gắn Career)  
3. **Streamlit** — giao diện người dùng  

> Model dự đoán chính vẫn là **Random Forest**.  
> Pipeline hiện tại: **dữ liệu thật (A+1)** — không augment Career skills, không refine centroid/Jaccard.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cấu trúc thư mục & từng file](#2-cấu-trúc-thư-mục--từng-file)
3. [Luồng hoạt động](#3-luồng-hoạt-động)
4. [Các kỹ thuật](#4-các-kỹ-thuật)
5. [Dữ liệu](#5-dữ-liệu)
6. [Artifacts `models/`](#6-artifacts-models)
7. [Cách chạy](#7-cách-chạy)
8. [Độ chính xác](#8-độ-chính-xác)
9. [Cấu hình môi trường](#9-cấu-hình-môi-trường)
10. [Tài liệu liên quan](#10-tài-liệu-liên-quan)

---

## 1. Tổng quan

### Bài toán

Người dùng mô tả bản thân (ngành học, sở thích, kỹ năng, dự án, thực tập, nghề thích/không thích). Hệ thống trả về **Top-5 nghề phù hợp** (UI nhấn **Top-3**), kèm % và giải thích.

### Kiến trúc logic

```text
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (train)                          │
│  CSV multi-industry                                             │
│    → OneHot Field + Soft/numeric + Skills cột CSV               │
│    → Random Forest                                              │
│    → Lưu models/*.pkl (RF + encoders + meta)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE (app)                             │
│  Bio → Gemini (JSON features)                                   │
│    → Vector (OneHot Field + numeric + skill binary từ vocab CSV)│
│    → RF.predict_proba → Preferred/Excluded                      │
│    → Top-5 + Gemini giải thích → Streamlit UI                   │
└─────────────────────────────────────────────────────────────────┘
```

### Stack

| Thành phần | Công nghệ |
|------------|-----------|
| UI | Streamlit |
| LLM | Google Gemini (`google-generativeai`) |
| ML | scikit-learn Random Forest |
| Encode | OneHotEncoder, LabelEncoder, MultiLabelBinarizer |
| Persist | joblib |
| Data | pandas, CSV |

---

## 2. Cấu trúc thư mục & từng file

```text
Career-Guidance-AI/
├── README.md                          ← tài liệu này
├── .venv/                             ← virtualenv (local, không commit)
└── career_ai_project/                 ← mã nguồn chính
    ├── app.py
    ├── model.py
    ├── train_model.py
    ├── evaluate.py
    ├── data_check.py
    ├── prompt_test.txt
    ├── requirements.txt
    ├── .env                           ← API key (không commit)
    ├── .gitignore
    ├── FIX_ACCURACY.md
    ├── IMPROVEMENTS.md
    ├── data/
    │   ├── career_prediction_multi_industry.csv
    │   └── career_prediction_IT_900rows.csv
    └── models/                        ← sinh bởi train_model.py
        ├── rf_model.pkl
        ├── field_encoder.pkl
        ├── career_encoder.pkl
        ├── skills_encoder.pkl
        └── model_meta.pkl
```

### 2.1. Thư mục gốc repo

| Mục | Ý nghĩa |
|-----|---------|
| `README.md` | Tài liệu tổng hợp dự án |
| `career_ai_project/` | Toàn bộ ứng dụng (code, data, models, docs kỹ thuật) |
| `.venv/` | Môi trường Python local |

### 2.2. Code Python (`career_ai_project/`)

#### `app.py` — Giao diện Streamlit

**Vai trò:** UI duy nhất người dùng tương tác.

**Hoạt động:**

1. Nhận đoạn bio (tối thiểu `MIN_TEXT_LENGTH` ký tự)  
2. `load_ml_artifacts()` — load RF + encoders + extras  
3. `call_gemini_extractor(...)` — trích JSON hồ sơ  
4. `validate_features(...)` — chuẩn hóa điểm / Skills / Preferred–Excluded  
5. `build_feature_vector(...)` → `predict_top_careers(...)`  
6. `explain_top_careers_with_gemini(...)` — giải thích từng nghề Top-k  
7. Hiển thị dashboard điểm kỹ năng + **Top-3 nổi bật** + Top-5 chi tiết  

**Chạy:** `streamlit run app.py`

---

#### `model.py` — Logic online (Gemini + inference ML)

**Vai trò:** Toàn bộ nghiệp vụ khi chạy app (không train).

| Nhóm hàm | Ý nghĩa |
|----------|---------|
| `build_extractor_prompt` | Prompt Gemini lần 1: Field, điểm 0–5, Skills, Preferred/Excluded, Reason |
| `call_gemini_extractor` | Gọi Gemini, parse JSON |
| `validate_features` | Ép kiểu, clamp điểm, lọc Skills/Career theo vocab hợp lệ |
| `load_ml_artifacts` | Load `models/*.pkl` (RF + 3 encoder + meta) |
| `get_field_classes` / `encode_field` | Đọc categories OneHot; map Field gần đúng |
| `build_feature_vector` | Tạo `X_global` (+ skill_list) khớp lúc train — Skills vocab CSV |
| `predict_top_careers` | RF proba → Preferred/Excluded → Top-k + % |
| `explain_top_careers_with_gemini` | Prompt Gemini lần 2: giải thích từng nghề |

**Lưu ý:** Phần `*_Reason` chỉ để hiển thị giải thích, **không** đưa vào RF.

---

#### `train_model.py` — Huấn luyện offline

**Vai trò:** Train Random Forest, đánh giá holdout, lưu artifacts.

**Các bước chính:**

1. Đọc `data/career_prediction_multi_industry.csv`  
2. Parse cột **Skills** (không gắn Career/JSON)  
3. Encode: Field OneHot, Career LabelEncoder, Skills MultiLabelBinarizer (vocab từ CSV)  
4. Đánh giá holdout 20% — RF thuần  
5. Train final trên toàn bộ dữ liệu  
6. Lưu `rf_model` + 3 encoder + `model_meta`  

**Chạy:** `python train_model.py`

**Model dự đoán:** Random Forest (`n_estimators=400`, `class_weight=balanced_subsample`).

---

#### `evaluate.py` — Kiểm thử LLM (Reliability / Bias)

**Vai trò:** Script thử Gemini nhiều lần cùng input (ổn định), hoặc input có yếu tố bias.

Không thay thế metric accuracy của RF trong `train_model.py`.

---

#### `data_check.py` — Kiểm tra CSV nhanh

In shape, missing, duplicate của `career_prediction_multi_industry.csv`.

---

#### `prompt_test.txt`

Mẫu câu / ghi chú dùng khi test prompt Gemini thủ công.

---

#### `requirements.txt`

```text
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
streamlit==1.42.2
google-generativeai==0.8.4
python-dotenv==1.0.1
numpy==2.2.3
```

---

#### Docs kỹ thuật trong project

| File | Nội dung |
|------|----------|
| `FIX_ACCURACY.md` | Fix accuracy khi đổi sang multi-industry (OneHot Field) |
| `IMPROVEMENTS.md` | Pipeline A+1: Skills CSV thật, RF thuần |

---

### 2.3. Thư mục `data/`

| File | Dùng để train? | Ý nghĩa |
|------|----------------|---------|
| **`career_prediction_multi_industry.csv`** | **Có — train chính** | ~9004 mẫu, ~239 Career, ~66 Field |
| **`career_prediction_IT_900rows.csv`** | **Không** | Dataset IT cũ — benchmark / tham chiếu |

---

### 2.4. Thư mục `models/` (sinh sau train)

Xem [mục 6](#6-artifacts-models).

---

## 3. Luồng hoạt động

### 3.1. Offline — Train

```text
multi_industry.csv
        │
        ├─► Parse cột Skills (CSV gốc)
        ├─► Field → OneHotEncoder
        ├─► Numeric (6 cột)
        ├─► Career → LabelEncoder
        │
        ▼
RandomForestClassifier.fit
        │
        ├─► rf_model.pkl
        ├─► field / career / skills encoders
        └─► model_meta.pkl (+ metrics holdout)
```

### 3.2. Online — App inference

```text
User bio
    │
    ▼
Gemini #1 (extractor)
    → Field, soft scores, Skills[], Preferred/Excluded
    │
    ▼
validate_features → build_feature_vector
    → X = [Field OneHot | numeric 6 | skill binary]
    │
    ▼
RF.predict_proba → Preferred/Excluded → Top-5 %
    │
    ├─► UI Top-3 + Top-5
    └─► Gemini #2 → giải thích
```

---

## 4. Các kỹ thuật

### 4.1. OneHot Encoding cho `Field`

- **Trước:** `LabelEncoder` → ~43% top-1 với 66 Field.
- **Sau:** `OneHotEncoder` → ~67% top-1 (Skills CSV, RF thuần).

### 4.2. Random Forest

- Phân loại đa lớp **Career** (~239 lớp).
- Input: OneHot Field + 6 numeric + skill multi-hot (vocab từ CSV).
- `n_estimators=400`, `class_weight=balanced_subsample`.

### 4.3. MultiLabelBinarizer cho Skills

Chuỗi `"Python, SQL, Git"` → vector binary theo vocabulary (~46 skill từ CSV).

### 4.4. Gemini LLM (2 lần gọi)

| Lần | Việc |
|-----|------|
| 1 | Trích JSON feature từ bio |
| 2 | Giải thích vì sao hồ sơ khớp từng nghề Top-k |

Gemini **không** thay RF chọn nghề.

---

## 5. Dữ liệu

### Schema CSV chính (`career_prediction_multi_industry.csv`)

| Cột | Kiểu | Vai trò |
|-----|------|---------|
| `Field` | string | Lĩnh vực (OneHot) |
| `Professional Skills` | 0–5 | Điểm chuyên môn |
| `Communication Skills` | 0–5 | Giao tiếp |
| `Problem Solving Skills` | 0–5 | Giải quyết vấn đề |
| `Teamwork Skills` | 0–5 | Làm việc nhóm |
| `Projects` | int ≥ 0 | Số dự án |
| `Internships` | int ≥ 0 | Số kỳ thực tập |
| `Career` | string | **Nhãn** cần dự đoán |
| `Skills` | string CSV | Kỹ năng/công nghệ cụ thể |

Quy mô: ~**9004** dòng, ~**239** Career, ~**66** Field.

---

## 6. Artifacts `models/`

| File | Nội dung |
|------|----------|
| `rf_model.pkl` | Random Forest đã train |
| `field_encoder.pkl` | OneHotEncoder cho Field |
| `career_encoder.pkl` | LabelEncoder cho tên Career |
| `skills_encoder.pkl` | MultiLabelBinarizer (vocab Skills CSV) |
| `model_meta.pkl` | metrics holdout |

Sinh bằng: `python train_model.py`. App **bắt buộc** có bộ file này trước khi chạy.

---

## 7. Cách chạy

```bash
cd career_ai_project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Tạo file .env
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-3.5-flash-lite

python train_model.py              # tạo / cập nhật models/
streamlit run app.py               # mở UI (thường http://localhost:8501)
```

### Lệnh phụ (tuỳ chọn)

```bash
python data_check.py               # thống kê CSV
python evaluate.py                 # test Reliability/Bias Gemini (cần API)
```

### Log train cần nhìn

```text
=== Đánh giá holdout (20%) — Skills CSV gốc, RF thuần ===
   top1=...  top3=...  top5=...
```

---

## 8. Độ chính xác

Holdout 20%, multi-industry (~239 nghề), pipeline **A+1**:

| Kịch bản | Top-1 | Top-3 | Top-5 |
|----------|-------|-------|-------|
| Skills CSV gốc (backup) | ~67% | ~94% | ~98% |
| **≥6 skills (giữ gốc + pad, ~55% có skill nghề)** | **~81%** | **~97%** | **~99%** |
| Giả lập Gemini kém (≤1 skill) | ~64–68% | ~90–93% | — |

Chi tiết: [`career_ai_project/IMPROVEMENTS.md`](career_ai_project/IMPROVEMENTS.md).

---

## 9. Cấu hình môi trường

File `career_ai_project/.env` (không commit):

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
```

`.gitignore` đã loại `.env`, `__pycache__`, v.v.

---

## 10. Tài liệu liên quan

| File | Khi nào đọc |
|------|-------------|
| [`career_ai_project/FIX_ACCURACY.md`](career_ai_project/FIX_ACCURACY.md) | Vì sao accuracy tụt khi đổi multi-industry & fix OneHot |
| [`career_ai_project/IMPROVEMENTS.md`](career_ai_project/IMPROVEMENTS.md) | Pipeline dữ liệu thật (A+1) |

---

## Tóm tắt một dòng

**Gemini trích hồ sơ → Random Forest (OneHot Field + Skills CSV) → Preferred/Excluded → Top-k nghề trên Streamlit.**
