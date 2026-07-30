# Career Guidance AI

Ứng dụng gợi ý hướng nghiệp bằng cách kết hợp Gemini LLM và mô hình Random Forest.
Người dùng nhập đoạn tự giới thiệu, hệ thống sẽ:
1. trích xuất các chỉ số kỹ năng từ văn bản,
2. dự đoán nghề nghiệp phù hợp,
3. hiển thị kết quả và giải thích bằng giao diện Streamlit.

## Cấu trúc dự án

```text
career_ai_project/
├── app.py                  # Giao diện Streamlit
├── model.py                # Logic xử lý: gọi Gemini, chuẩn hóa dữ liệu, dự đoán nghề nghiệp
├── train_model.py          # Huấn luyện mô hình Random Forest và lưu artifact
├── data/
│   └── career_prediction.csv
├── models/
│   ├── rf_model.pkl
│   ├── field_encoder.pkl
│   └── career_encoder.pkl
├── .env                    # Chứa GEMINI_API_KEY và GEMINI_MODEL (nếu dùng)
└── requirements.txt        # Danh sách dependency
```

## Vai trò từng file

- `app.py`
  - Chịu trách nhiệm giao diện người dùng bằng Streamlit.
  - Nhận đoạn văn giới thiệu từ người dùng.
  - Gọi các hàm từ `model.py` để phân tích và hiển thị kết quả.

- `model.py`
  - Chứa toàn bộ logic nghiệp vụ của hệ thống.
  - Gọi Gemini để trích xuất các đặc trưng kỹ năng và lý do giải thích.
  - Chuẩn hóa dữ liệu đầu vào cho mô hình ML.
  - Dự đoán top nghề nghiệp bằng Random Forest.

- `train_model.py`
  - Huấn luyện mô hình Random Forest từ dữ liệu trong `data/career_prediction.csv`.
  - Lưu các file `.pkl` vào thư mục `models/` để dùng cho inference.

- `data/career_prediction.csv`
  - Tập dữ liệu dùng để huấn luyện mô hình.

- `models/`
  - Chứa các artifact đã được train: model và encoder.

## Luồng hoạt động

1. Người dùng nhập đoạn mô tả bản thân vào giao diện Streamlit.
2. `app.py` gửi nội dung này sang `model.py`.
3. `model.py` dùng Gemini để suy ra các feature như:
   - Field
   - Coding Skills
   - Communication Skills
   - Problem Solving Skills
   - Teamwork Skills
   - Projects
   - Internships
4. Các giá trị này được chuẩn hóa và đưa vào mô hình Random Forest.
5. Hệ thống trả về Top 5 nghề phù hợp nhất, kèm giải thích ngắn gọn.


## Cách chạy

```bash
cd career_ai_project
py -m streamlit run app.py
```

# (khuyến nghị) tạo venv

python3 -m venv .venv
source .venv/bin/activate

# dependencies

pip install: streamlit, pandas, scikit-learn, google.generativeai,...

pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
streamlit==1.42.2
google-generativeai==0.8.4
python-dotenv==1.0.1
numpy==2.2.3

# chạy app

streamlit run app.py

## Tính năng đang phát triển 

1. Viết bộ test riêng cho từng trục đánh giá:
   - Reliability: chạy nhiều lần cùng input, kiểm tra kết quả ổn định
   - Bias: thử các mô tả có yếu tố giới tính/vùng miền khác nhau nhưng cùng nội dung chuyên môn, xem % phù hợp có lệch bất thường không
   - Robustness: thử thay đổi nhẹ câu chữ (từ đồng nghĩa, lỗi chính tả nhẹ) xem kết quả có ổn định không
   - Explainability: đã có sẵn phần "Vì sao gợi ý" trong app, có thể mở rộng thêm
   - Social Impact: đánh giá định tính về tác động của việc gợi ý sai ngành nghề
2. Sử dụng OCR cho phép sinh viên upload bảng điểm dự đoán
