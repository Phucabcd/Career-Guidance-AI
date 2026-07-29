cd career_ai_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 1. Điền Gemini API key vào .env
# GEMINI_API_KEY=...
# 2. Train (đã chạy sẵn; chạy lại nếu đổi CSV)
python train_model.py
# 3. Mở app
streamlit run app.py