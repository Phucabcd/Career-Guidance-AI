# Career Guidance AI

Ứng dụng gợi ý nghề nghiệp từ đoạn tự giới thiệu của người dùng. Hệ thống kết hợp **Gemini** để trích xuất thông tin từ văn bản và tạo giải thích, **Random Forest** để xếp hạng nghề nghiệp, và **Streamlit** cho giao diện web.

## Luồng hoạt động

```text
Người dùng nhập mô tả bản thân
            │
            ▼
Gemini trích xuất Field, kỹ năng, dự án, thực tập,
điểm kỹ năng và nghề thích/không thích
            │
            ▼
Chuẩn hóa thành vector đặc trưng
            │
            ▼
Random Forest dự đoán Top 5 nghề
            │
            ▼
Gemini tạo giải thích → Streamlit hiển thị kết quả
```

## Cấu trúc dự án

```text
Tu_duy_TTNT/
├── app.py                              # Giao diện Streamlit
├── model.py                            # Gemini, chuẩn hóa, dự đoán và giải thích
├── train_model.py                      # Huấn luyện, đánh giá, lưu model artifacts
├── data_check.py                       # Script kiểm tra dữ liệu đơn giản
├── requirements.txt                    # Dependencies và phiên bản sử dụng
├── .env                                # GEMINI_API_KEY / GEMINI_MODEL (không commit)
├── data/
│   ├── career_prediction_multi_industry.csv  # Dataset train chính hiện tại
│   ├── career_prediction_IT_900rows.csv      # Dataset IT tham khảo
│   └── career_prediction.csv                 # Dataset cũ/nhỏ
└── models/
    ├── rf_model.pkl                    # Random Forest đã huấn luyện
    ├── field_encoder.pkl               # OneHotEncoder cho Field
    ├── career_encoder.pkl              # LabelEncoder cho Career
    ├── skills_encoder.pkl              # MultiLabelBinarizer cho Skills
    └── model_meta.pkl                  # Metrics monitoring của lần train cuối
```

`.venv/` và `__pycache__/` được sinh trong quá trình chạy. Không cần tự chỉnh sửa các file `.pkl` trong `models/`.

## Dữ liệu và đặc trưng

`train_model.py` hiện đọc **`data/career_prediction_multi_industry.csv`**. Dataset cần có schema:

```text
Field, Professional Skills, Communication Skills, Problem Solving Skills,
Teamwork Skills, Projects, Internships, Career, Skills
```

Mỗi hồ sơ được mã hóa theo thứ tự:

```text
[Field one-hot | 4 điểm kỹ năng | Projects | Internships | Skills multi-hot]
```

- `Field`: one-hot encoding, không áp đặt thứ tự số học giữa các lĩnh vực.
- Bốn điểm kỹ năng: thang 0–5.
- `Skills`: danh sách kỹ năng/công nghệ phân cách bằng dấu phẩy trong CSV; được multi-hot encoding.
- `Career`: nhãn đích mà mô hình học để dự đoán.

## Vai trò các module

### `app.py`

Điểm khởi động của web app. Nhận đoạn giới thiệu, gọi pipeline trong `model.py`, và hiển thị dashboard hồ sơ, điểm kỹ năng, số dự án/thực tập và Top 5 nghề.

### `model.py`

Đảm nhiệm inference theo các bước:

1. Load Random Forest và encoder từ `models/`.
2. Gọi Gemini lần một để rút trích `Field`, bốn điểm kỹ năng, `Skills`, `Projects`, `Internships`, `Preferred_Careers` và `Excluded_Careers`.
3. Kiểm tra, chuẩn hóa dữ liệu Gemini theo vocabulary đã học từ dataset.
4. Tạo feature vector đúng thứ tự lúc train và lấy xác suất từ Random Forest.
5. Loại nghề người dùng không muốn; tăng trọng số cho nghề được ưu tiên.
6. Lấy Top 5 và chuẩn hóa chúng thành tỷ lệ hiển thị. Các tỷ lệ này cộng xấp xỉ 100% *trong Top 5*, không phải xác suất tuyệt đối trên toàn bộ nghề.
7. Gọi Gemini lần hai để viết lời giải thích tiếng Việt cho từng gợi ý.

### `train_model.py`

Huấn luyện Random Forest từ dataset chính. Script kiểm tra/làm sạch dữ liệu, mã hóa đặc trưng, chia train/test để monitoring, rồi theo dõi Top-1/3/5 accuracy, robustness gap khi thêm nhiễu, Brier score và fairness gap theo `Field`. Cuối cùng model được huấn luyện trên 100% dữ liệu và lưu vào `models/`.

## Cài đặt và chạy

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

Tạo `.env` tại thư mục gốc:

```env
GEMINI_API_KEY=your_api_key_here
# Có thể bỏ qua để dùng giá trị mặc định trong model.py
GEMINI_MODEL=gemini-3.5-flash-lite
```

Huấn luyện lại model khi thay đổi dataset hoặc feature engineering:

```powershell
python train_model.py
```

Khởi động giao diện:

```powershell
streamlit run app.py
```

## Lưu ý vận hành

- Cần có `rf_model.pkl`, `field_encoder.pkl`, `career_encoder.pkl` và `skills_encoder.pkl` trước khi mở app. Nếu thiếu, chạy `python train_model.py`.
- Gemini là thành phần bắt buộc để chuyển văn bản tự do thành feature vector. Thiếu hoặc sai API key thì không thể dự đoán từ mô tả người dùng.
- Nếu model Gemini không khả dụng hoặc hết quota, đổi `GEMINI_MODEL` trong `.env` sang một model mà API key được phép sử dụng.
- Không commit `.env` vì file chứa khóa bí mật.
- `data_check.py` là script cũ; đường dẫn của nó cần được sửa trước khi dùng trong cấu trúc project hiện tại.

## Dependencies

Xem phiên bản chính xác trong [`requirements.txt`](requirements.txt): pandas, NumPy, scikit-learn, joblib, Streamlit, Google Generative AI và python-dotenv.
