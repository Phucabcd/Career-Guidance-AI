# Career Guidance AI - demo app

## Cấu trúc
- `model.py` — logic gợi ý ngành nghề (embedding + cosine similarity + trích từ khóa khớp)
- `app.py` — giao diện Streamlit gọi vào `model.py`
- `requirements.txt` — thư viện cần cài

## Cách chạy

```bash
py -m streamlit run app.py
```

## Bước tiếp theo gợi ý
1. Mở rộng `NGANH_NGHE` trong `model.py` với nhiều ngành hơn, mô tả chi tiết hơn.
2. Viết bộ test riêng cho từng trục đánh giá:
   - Reliability: chạy nhiều lần cùng input, kiểm tra kết quả ổn định
   - Bias: thử các mô tả có yếu tố giới tính/vùng miền khác nhau nhưng cùng nội dung chuyên môn, xem % phù hợp có lệch bất thường không
   - Robustness: thử thay đổi nhẹ câu chữ (từ đồng nghĩa, lỗi chính tả nhẹ) xem kết quả có ổn định không
   - Explainability: đã có sẵn phần "Vì sao gợi ý" trong app, có thể mở rộng thêm
   - Social Impact: đánh giá định tính về tác động của việc gợi ý sai ngành nghề


## Công nghệ sử dụng & các syntax
streamlit  --> UI 
sentence-transformers --> 

- similarities: là mảng các con số đại diện cho độ giống nhau.
- enumerate: là một hàm có sẵn của py dùng để lấy lập thêm index (i)