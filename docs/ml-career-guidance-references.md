# Tài liệu tham khảo: AI Hướng nghiệp / Career Guidance

Tài liệu này tập trung vào **chủ đề** và **yêu cầu** của dự án Career Guidance AI, kèm **kỹ thuật nên dùng** và **link tham khảo**. Không mô tả chi tiết triển khai hiện tại trong code.

---

## 1. Chủ đề dự án

Hệ thống AI hỗ trợ **hướng nghiệp / gợi ý ngành nghề** cho học sinh–sinh viên, dựa trên hồ sơ cá nhân (sở thích, điểm mạnh, kỹ năng quan tâm).

Đây là bài toán thuộc nhóm:

- Career Recommendation
- Student Career Prediction
- AI for Education / Career Counselling

---

## 2. Yêu cầu dự án

1. Sử dụng / mở rộng dữ liệu nghề–kỹ năng (ví dụ dataset Mendeley bên dưới).
2. Nhận mô tả của học sinh → **gợi ý ngành phù hợp** kèm mức độ khớp.
3. Có giao diện demo để người dùng tương tác.
4. Đánh giá hệ thống theo **5 trục**:
   - **Reliability**: cùng input chạy nhiều lần → kết quả ổn định?
   - **Bias**: đổi yếu tố giới tính / vùng miền (giữ nội dung chuyên môn) → % khớp có lệch bất thường?
   - **Robustness**: paraphrase, lỗi chính tả nhẹ → ranking có ổn?
   - **Explainability**: giải thích được vì sao gợi ý ngành đó.
   - **Social Impact**: đánh giá định tính rủi ro khi gợi ý sai ngành.
5. Ưu tiên AI **hỗ trợ quyết định**, không thay thế hoàn toàn tư vấn viên.

---

## 3. Kỹ thuật nên dùng

Các kỹ thuật dưới đây phù hợp với chủ đề hướng nghiệp (không bắt buộc dùng hết; nên chọn theo dữ liệu và phạm vi đề tài).

| Mục tiêu | Kỹ thuật nên cân nhắc | Ghi chú |
|---|---|---|
| Gợi ý nghề khi học sinh mới (ít lịch sử tương tác) | **Content-based filtering** | Phù hợp cold-start |
| Input là văn mô tả tự do | **TF-IDF + cosine** (baseline) và/hoặc **Sentence embeddings + similarity** (SBERT) | So khớp profile ↔ mô tả nghề/kỹ năng |
| Có nhãn “học sinh → ngành đúng” | **Classification**: Random Forest, SVM, XGBoost, Neural Network | Thường gặp trong survey career prediction |
| Có nhiều user / tương tác | **Collaborative filtering** hoặc **Hybrid** (content + collaborative) | Tăng cá nhân hóa |
| Cần giải thích gợi ý | Keyword overlap, feature importance, mapping RIASEC, XAI đơn giản | Đáp ứng Explainability |
| Reliability / Robustness | Behavioral testing (paraphrase, typo, rerun) | Khớp 5 trục đánh giá |
| Bias | Counterfactual tests + fairness checks | Tránh lệch nhóm người dùng |
| Social Impact | Human-in-the-loop, disclaimer, hybrid AI + tư vấn | Giảm rủi ro gợi ý sai |

### Lộ trình kỹ thuật gợi ý

1. **Baseline:** TF-IDF + cosine similarity trên mô tả nghề/kỹ năng.
2. **Main model:** Content-based với dense embeddings (ưu tiên multilingual nếu input tiếng Việt).
3. **Tuỳ chọn nâng cao:** Classifier (RF / XGBoost / SVM) nếu có nhãn; hoặc hybrid recommender.
4. **Explainability:** trả về skill/từ khóa khớp + lý do ngắn.
5. **Evaluation suite:** test theo 5 trục Reliability / Bias / Robustness / Explainability / Social Impact.
6. **Tuỳ chọn:** Chatbot generative chỉ khi đã có retrieval + guardrail (tránh automation bias).

### Bộ kỹ thuật ưu tiên cho đề tài này

1. Content-based recommendation  
2. Text representation: TF-IDF (baseline) + Sentence embeddings (chính)  
3. Cosine similarity / Top-k ranking  
4. (Nếu có nhãn) Random Forest / XGBoost / SVM  
5. Explainable matching (keyword / feature rationale)  
6. Fairness & robustness testing  

---

## 4. Link tham khảo

### 4.1. Survey / tổng quan AI hướng nghiệp

- Frontiers (2026) — Implementation of AI in career counselling for university students: a systematic review  
  https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1787689/full

- JOTSE — Artificial intelligence in education: A systematic literature review of machine learning approaches in student career prediction  
  https://www.jotse.org/index.php/jotse/article/view/3124/937  
  DOI: https://doi.org/10.3926/jotse.3124

- SLR — Application of Artificial Intelligence in Career Counselling  
  https://doi.org/10.51889/2959-5762.2025.85.1.004

- IJCSE — AI powered career recommendation for students (survey)  
  https://www.ijcsejournal.org/ai-powered-career-recommendation/

### 4.2. Lý thuyết hướng nghiệp (nền tảng lĩnh vực)

- Holland Codes (RIASEC) — Person–Environment Fit  
  https://en.wikipedia.org/wiki/Holland_Codes

- O\*NET Interest Profiler  
  https://www.onetcenter.org/IP.html

- UNESCO — Career guidance  
  https://www.unesco.org/en/tags/career-guidance

### 4.3. Recommender systems

- Google ML — Content-based filtering  
  https://developers.google.com/machine-learning/recommendation/content-based/basics

- Google ML — Collaborative filtering  
  https://developers.google.com/machine-learning/recommendation/collaborative/basics

- Mining of Massive Datasets — Recommendation Systems  
  http://www.mmds.org/

### 4.4. NLP / semantic matching

- Sentence-BERT (Reimers & Gurevych, 2019)  
  https://arxiv.org/abs/1908.10084

- Sentence-Transformers documentation  
  https://www.sbert.net/

- Hugging Face NLP Course  
  https://huggingface.co/learn/nlp-course

### 4.5. Đánh giá & Responsible AI (khớp 5 trục)

- Fairness — Google ML Crash Course  
  https://developers.google.com/machine-learning/crash-course/fairness

- Checklist: Beyond Accuracy (NLP behavioral testing)  
  https://arxiv.org/abs/2004.05465

- Interpretable Machine Learning (Christoph Molnar)  
  https://christophm.github.io/interpretable-ml-book/

- NIST AI Risk Management Framework  
  https://www.nist.gov/itl/ai-risk-management-framework

### 4.6. Dataset

- Career dataset (Mendeley) — đang dùng trong dự án  
  https://data.mendeley.com/datasets/4spj4mbpjr/2

- O\*NET databases  
  https://www.onetcenter.org/database.html

- ESCO — European Skills, Competences, Qualifications and Occupations  
  https://esco.ec.europa.eu/en

---

## 5. Mapping kỹ thuật → đánh giá 5 trục

| Trục | Câu hỏi kiểm tra | Gợi ý cách làm |
|---|---|---|
| Reliability | Cùng input nhiều lần có cùng ranking? | Rerun cố định seed / model; so sánh Top-k |
| Bias | Đổi giới tính/vùng miền có lệch % khớp? | Counterfactual prompts; đo delta score |
| Robustness | Đồng nghĩa / typo nhẹ có đảo thứ hạng mạnh? | Paraphrase set + typo set |
| Explainability | Người dùng hiểu vì sao được gợi ý? | Hiện skill khớp, lý do ngắn, feature highlights |
| Social Impact | Gợi ý sai gây hại thế nào? | Case review định tính + disclaimer + human review |

---

## 6. Gợi ý trích dẫn ngắn cho báo cáo

Khi viết báo cáo / README học thuật, có thể nhóm tài liệu như sau:

1. **Bối cảnh AI hướng nghiệp:** Frontiers 2026; JOTSE SLR career prediction.  
2. **Phương pháp gợi ý:** Google ML Recommendation (content / collaborative); MMDS Ch. Recommendation.  
3. **Biểu diễn văn bản:** Sentence-BERT; Hugging Face NLP Course.  
4. **Đánh giá có trách nhiệm:** Google Fairness; Checklist Beyond Accuracy; NIST AI RMF.  
5. **Dữ liệu nghề nghiệp:** Mendeley Career Dataset; O\*NET; ESCO.
