# 🧭 Career Guidance AI — Hệ thống Gợi ý Nghề nghiệp

[**Tiếng Việt**](#tiếng-việt) | [**English**](#english)

---

<a name="tiếng-việt"></a>
## 🇻🇳 Tiếng Việt

Ứng dụng gợi ý nghề nghiệp thông minh kết hợp giữa **LLM (Gemini)** và **Machine Learning (Random Forest)**. 

Hệ thống sử dụng Gemini để trích xuất thông tin cấu trúc từ đoạn tự giới thiệu tự do của người dùng, đưa qua Random Forest đã huấn luyện để xếp hạng Top 5 nghề nghiệp phù hợp nhất, và cuối cùng dùng Gemini tạo giải thích chi tiết cho từng gợi ý trên giao diện **Streamlit**.

### 🔄 Luồng hoạt động

```text
Người dùng nhập mô tả bản thân
            │
            ▼
Gemini trích xuất Field, 4 điểm kỹ năng, dự án, thực tập,
kỹ năng chuyên môn, nghề thích/không thích & lý do
            │
            ▼
Chuẩn hóa dữ liệu & chuyển thành Feature Vector
            │
            ▼
Random Forest dự đoán xác suất cho các nghề nghiệp
            │
            ▼
Điều chỉnh trọng số (ưu tiên/loại trừ nghề) -> Top 5
            │
            ▼
Gemini tạo giải thích tiếng Việt -> Streamlit hiển thị
```

### 📁 Cấu trúc dự án

```text
Tu_duy_TTNT/
├── app.py                              # Giao diện Streamlit & xử lý tương tác người dùng
├── model.py                            # Gemini API, chuẩn hóa vector, dự đoán RF & sinh giải thích
├── train_model.py                      # Pipeline huấn luyện RF, đánh giá monitoring & xuất artifacts
├── prompt_test.txt                     # Bộ prompt mẫu & test case thực tế
├── style.css                           # Custom CSS cho giao diện Streamlit
├── requirements.txt                    # Danh sách thư viện cần thiết
├── .env                                # Chứa GEMINI_API_KEY / GEMINI_MODEL (không commit)
├── data/
│   ├── career_prediction_multi_industry.csv  # Dataset huấn luyện chính (Đa ngành)
│   ├── career_prediction_IT_900rows.csv      # Dataset chuyên IT (tham khảo)
│   ├── career_prediction.csv                 # Dataset cũ/nhỏ
│   └── README.txt                            # Mô tả chi tiết về các dataset
└── models/
    ├── rf_model.pkl                    # Model Random Forest đã huấn luyện
    ├── field_encoder.pkl               # OneHotEncoder cho thuộc tính Field
    ├── career_encoder.pkl              # LabelEncoder cho nhãn Career
    ├── skills_encoder.pkl              # MultiLabelBinarizer cho danh sách Skills
    └── model_meta.pkl                  # Chỉ số đánh giá (metrics monitoring) của model
```

*Lưu ý: `.venv/` và `__pycache__/` được sinh tự động. Không cần tự sửa các file `.pkl` trong `models/`.*

### 📊 Dữ liệu và đặc trưng

Mô hình huấn luyện dựa trên file `data/career_prediction_multi_industry.csv` với các thuộc tính:
- `Field`: Lĩnh vực/Ngành học (mã hóa One-Hot).
- 4 Điểm kỹ năng (thang 0–5): `Professional Skills`, `Communication Skills`, `Problem Solving Skills`, `Teamwork Skills`.
- `Projects` & `Internships`: Số lượng dự án thực tế và kỳ thực tập.
- `Skills`: Danh sách kỹ năng công nghệ/chuyên môn (mã hóa Multi-Hot).
- `Career`: Nhãn vị trí công việc đích (Target).

Thứ tự Feature Vector chuẩn hóa:
```text
[Field (One-Hot) | 4 điểm Kỹ năng | Projects | Internships | Skills (Multi-Hot)]
```

### 🧩 Vai trò các module

- **`app.py`**: Khởi chạy Streamlit UI, hiển thị form nhập mô tả, nút tải prompt mẫu, hiển thị radar chart kỹ năng, danh sách nghề thích/ghét và Top 5 gợi ý nghề kèm giải thích.
- **`model.py`**: Quản lý toàn bộ logic inference. Gọi Gemini trích xuất thông tin JSON, validate & fill default nếu lỗi format, xây dựng feature vector, dự đoán với Random Forest, áp dụng bộ lọc ưu tiên/loại trừ, và gọi Gemini sinh giải thích.
- **`train_model.py`**: Thực hiện chuẩn hóa dữ liệu, mã hóa đặc trưng, đánh giá model in-training (Top-1/3/5 accuracy, Robustness Gap khi thêm nhiễu, Brier Score, Fairness Gap theo Field), huấn luyện model trên 100% dữ liệu và lưu vào `models/`.

### 🛠️ Cài đặt và Chạy

1. **Khởi tạo môi trường ảo**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

2. **Cài đặt dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Cấu hình file `.env`**:
   Tạo file `.env` tại thư mục gốc với nội dung:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

4. **Huấn luyện lại mô hình (nếu cần)**:
   ```powershell
   python train_model.py
   ```

5. **Chạy ứng dụng**:
   ```powershell
   streamlit run app.py
   ```

### ⚠️ Lưu ý vận hành

- Cần đảm bảo thư mục `models/` có đầy đủ 4 file `.pkl` trước khi chạy app. Nếu thiếu, chạy `python train_model.py`.
- Gemini API key là bắt buộc để chuyển đổi mô tả văn bản thành vector đặc trưng.
- File `.env` chứa chìa khóa bí mật, tuyệt đối **không commit** lên Git (đã được cấu hình trong `.gitignore`).

---

<a name="english"></a>
## 🇬🇧 English

An intelligent career recommendation application combining **LLMs (Gemini)** and **Machine Learning (Random Forest)**.

The system uses Gemini to extract structured features from free-form user self-introductions, feeds them into a trained Random Forest classifier to rank the Top 5 most suitable careers, and generates detailed personalized explanations using Gemini within a **Streamlit** web interface.

### 🔄 System Workflow

```text
User enters self-description text
            │
            ▼
Gemini extracts Field, 4 Skill scores, Projects, Internships,
Professional Skills, Preferred/Excluded Careers & reasons
            │
            ▼
Data validation & Feature Vector construction
            │
            ▼
Random Forest predicts career probability distribution
            │
            ▼
Weight adjustments (preferred/excluded careers) -> Top 5
            │
            ▼
Gemini generates explanations -> Streamlit displays results
```

### 📁 Project Structure

```text
Tu_duy_TTNT/
├── app.py                              # Streamlit UI & user interaction handler
├── model.py                            # Gemini API integration, vector encoding, RF inference & explanations
├── train_model.py                      # RF model training pipeline, monitoring evaluation & artifact export
├── prompt_test.txt                     # Sample test prompts & real-world test cases
├── style.css                           # Custom CSS for Streamlit interface
├── requirements.txt                    # Python dependencies
├── .env                                # Contains GEMINI_API_KEY / GEMINI_MODEL (do not commit)
├── data/
│   ├── career_prediction_multi_industry.csv  # Main training dataset (Multi-industry)
│   ├── career_prediction_IT_900rows.csv      # IT-focused dataset (reference)
│   ├── career_prediction.csv                 # Legacy/small dataset
│   └── README.txt                            # Detailed description of datasets
└── models/
    ├── rf_model.pkl                    # Trained Random Forest model
    ├── field_encoder.pkl               # OneHotEncoder for Field feature
    ├── career_encoder.pkl              # LabelEncoder for Career target labels
    ├── skills_encoder.pkl              # MultiLabelBinarizer for Skills list
    └── model_meta.pkl                  # Model monitoring evaluation metrics
```

*Note: `.venv/` and `__pycache__/` are auto-generated. Do not manually modify `.pkl` files inside `models/`.*

### 📊 Data & Feature Representation

The model is trained on `data/career_prediction_multi_industry.csv` with the following feature schema:
- `Field`: Academic discipline or industry sector (One-Hot Encoded).
- 4 Skill Scores (0–5 scale): `Professional Skills`, `Communication Skills`, `Problem Solving Skills`, `Teamwork Skills`.
- `Projects` & `Internships`: Count of practical projects and completed internships.
- `Skills`: List of technical and domain skills (Multi-Hot Encoded).
- `Career`: Target job role label.

Normalized Feature Vector ordering:
```text
[Field (One-Hot) | 4 Skill Scores | Projects | Internships | Skills (Multi-Hot)]
```

### 🧩 Module Responsibilities

- **`app.py`**: Main application entry point. Renders the Streamlit dashboard, input text box, prompt template loader, skill radar chart, career preference summary, and Top 5 recommendations with AI explanations.
- **`model.py`**: Core inference engine. Handles Gemini JSON extraction, schema validation & fallback filling, feature vector transformation, Random Forest prediction, post-processing filters, and Gemini explanation prompting.
- **`train_model.py`**: Data preprocessing, feature encoding, in-training monitoring (Top-1/3/5 accuracy, Noise Robustness Gap, Brier Reliability Score, Group Fairness Gap across Fields), final model fitting on 100% data, and artifact export to `models/`.

### 🛠️ Installation & Setup

1. **Create and activate virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure `.env`**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

4. **Train/Retrain Model (optional)**:
   ```powershell
   python train_model.py
   ```

5. **Run the Application**:
   ```powershell
   streamlit run app.py
   ```

### ⚠️ Operational Notes

- Ensure `models/` contains all 4 required `.pkl` files before launching the app. Run `python train_model.py` if any are missing.
- A valid Gemini API Key is required for extracting structured feature vectors from text input.
- Keep `.env` confidential and **do not commit** it to version control (pre-configured in `.gitignore`).

---

### 📦 Dependencies

Defined in [`requirements.txt`](requirements.txt): `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `google-generativeai`, `python-dotenv`.
