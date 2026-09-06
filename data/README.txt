========================================================================
HƯỚNG DẪN DỮ LIỆU & MỞ RỘNG MÔ HÌNH (DATA & MODEL EXTENSION GUIDE)
========================================================================

File dữ liệu chính: data/career_prediction_multi_industry.csv 
Vì file quá lớn nên chúng tôi không thể push lên github

------------------------------------------------------------------------
1. CÁC CỘT TRONG CSV HIỆN TẠI:
------------------------------------------------------------------------
- Field: Lĩnh vực / Ngành học (Information Technology, Business & Economics,...)
- Professional Skills: Điểm kỹ năng chuyên môn (0 - 5)
- Communication Skills: Điểm kỹ năng giao tiếp (0 - 5)
- Problem Solving Skills: Điểm kỹ năng giải quyết vấn đề (0 - 5)
- Teamwork Skills: Điểm kỹ năng làm việc nhóm (0 - 5)
- Projects: Số dự án thực tế
- Internships: Số kỳ thực tập
- Skills: Các công nghệ/kỹ năng cụ thể (dấu phẩy phân cách, ví dụ: Python, Docker, SQL)
- Career: Nhãn nghề nghiệp dự đoán (TARGET / OUTPUT)

------------------------------------------------------------------------
2. TRƯỜNG HỢP 1: THÊM HÀNG DỮ LIỆU MỚI (Giữ nguyên cấu trúc cột)
------------------------------------------------------------------------
- Bổ sung các mẫu dữ liệu mới vào file CSV.
- Chạy lại lệnh huấn luyện:
    python train_model.py
- train_model.py sẽ tự động cập nhật các Encoder và fit lại mô hình Random Forest.

------------------------------------------------------------------------
3. TRƯỜNG HỢP 2: THÊM / SỬA CỘT THUỘC TÍNH (Mở rộng Input/Output)
------------------------------------------------------------------------
Bước 1: Cập nhật cột mới trong file CSV (ví dụ: thêm cột điểm Tiếng Anh English Skills).
Bước 2: Chỉnh sửa file `train_model.py`:
  - Thêm tên cột vào NUMERIC_FEATURE_COLUMNS (nếu là cột số) hoặc điều chỉnh encode_features() (nếu là cột chữ/categorical).
  - Nếu thay đổi cột nhãn đầu ra, đổi biến TARGET_COLUMN.
Bước 3: Cập nhật file `model.py` & `app.py`:
  - Cập nhật FEATURE_COLUMNS trong model.py khớp với train_model.py.
  - Cập nhật Prompt trích xuất của Gemini (build_extractor_prompt) để Gemini trích xuất thêm thuộc tính mới từ text tự giới thiệu của người dùng.
  - Cập nhật build_feature_vector() để xếp vector X đúng thứ tự.
Bước 4: Chạy lệnh huấn luyện lại:
    python train_model.py
