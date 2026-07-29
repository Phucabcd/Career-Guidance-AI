"""
app.py
Career Guidance AI — Streamlit app kết hợp:
  Phase 1: Gemini LLM trích xuất đặc trưng + lý do (Reasoning)
  Phase 3: Random Forest dự đoán nghề nghiệp từ các đặc trưng số

Chạy (sau khi đã train model):
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn & hằng số
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Thứ tự features BẮT BUỘC khớp với train_model.py (chỉ giá trị số / Field encode)
FEATURE_COLUMNS = [
    "Field",
    "Coding Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]

# Mapping kỹ năng → key lý do trong JSON Gemini (dùng cho UI)
SKILL_REASON_KEYS = {
    "Coding Skills": "Coding_Reason",
    "Communication Skills": "Communication_Reason",
    "Problem Solving Skills": "Problem_Solving_Reason",
    "Teamwork Skills": "Teamwork_Reason",
}

# Nhãn hiển thị cho từng kỹ năng trên dashboard
SKILL_UI_LABELS = {
    "Coding Skills": "💻 Kỹ năng Lập trình (Coding Skills)",
    "Communication Skills": "🗣️ Kỹ năng Giao tiếp (Communication Skills)",
    "Problem Solving Skills": "🧩 Kỹ năng Giải quyết vấn đề (Problem Solving Skills)",
    "Teamwork Skills": "🤝 Kỹ năng Làm việc nhóm (Teamwork Skills)",
}

# Độ dài tối thiểu của đoạn tự giới thiệu (ký tự, sau strip)
MIN_TEXT_LENGTH = 40


def build_extractor_prompt(user_text: str, list_of_all_careers: list[str]) -> str:
    """
    Prompt Gemini lần 1: chấm điểm kỹ năng + deep reasoning + Excluded_Careers.
    list_of_all_careers lấy từ career_encoder.classes_ để Gemini dùng đúng tên class.
    """
    careers_json = json.dumps(list_of_all_careers, ensure_ascii=False)

    return f"""
Bạn là bộ trích xuất đặc trưng hướng nghiệp (NLP Feature Extractor) kèm phân tích sâu (Deep Reasoning).
Nhiệm vụ: Đọc đoạn văn tự giới thiệu của người dùng, chấm điểm kỹ năng, phân tích lý do thật chi tiết,
VÀ phát hiện các ngành nghề bị HẠN CHẾ / KHÔNG THÍCH / TỪ CHỐI.

Danh sách ngành nghề hợp lệ trong hệ thống (phải dùng ĐÚNG tên này khi điền Excluded_Careers):
{careers_json}

QUY TẮC BẮT BUỘC:
1. Chỉ trả về JSON thuần túy — KHÔNG bọc trong markdown, KHÔNG thêm giải thích ngoài JSON, KHÔNG dùng ```json.
2. JSON phải có ĐÚNG các keys sau (không thiếu, không đổi tên):
   - "Field": string (một trong: IT, Business, Engineering, Design, Science; nếu không rõ hãy chọn gần nhất)
   - "Projects": integer (>= 0) — ước lượng số dự án đã làm
   - "Internships": integer (>= 0) — ước lượng số kỳ thực tập
   - "Coding Skills": integer từ 0 đến 4
   - "Coding_Reason": string (phân tích lý do điểm Coding Skills)
   - "Communication Skills": integer từ 0 đến 4
   - "Communication_Reason": string (phân tích lý do điểm Communication Skills)
   - "Problem Solving Skills": integer từ 0 đến 4
   - "Problem_Solving_Reason": string (phân tích lý do điểm Problem Solving Skills)
   - "Teamwork Skills": integer từ 0 đến 4
   - "Teamwork_Reason": string (phân tích lý do điểm Teamwork Skills)
   - "Excluded_Careers": array of string — các ngành bị hạn chế/không thích/từ chối
3. Thang điểm kỹ năng 0-4:
   0 = không đề cập / rất yếu
   1 = cơ bản
   2 = trung bình
   3 = khá
   4 = thành thạo / nổi bật trong văn bản
4. Đối với mỗi kỹ năng, hãy viết phần phân tích lý do (Reasoning) THẬT CHI TIẾT (khoảng 3-4 câu dài). Phải phân tích sâu sắc dựa trên câu chữ của ứng viên, trích dẫn lại ý của ứng viên để chứng minh, và đưa ra nhận xét ngắn về việc kỹ năng này ảnh hưởng thế nào đến thái độ làm việc của họ.
5. Ngoài việc chấm điểm kỹ năng, hãy phân tích xem người dùng có HẠN CHẾ, KHÔNG THÍCH, hoặc TỪ CHỐI ngành nghề nào không. Nếu có, hãy đối chiếu với danh sách các ngành nghề hợp lệ sau đây: {careers_json} và trả về một danh sách (mảng) tên các ngành đó trong "Excluded_Careers". Chỉ dùng đúng tên trong danh sách hợp lệ. Nếu không ghét ngành nào thì trả về mảng rỗng [].
6. Mỗi trường *_Reason phải dựa trên bằng chứng trong đoạn văn (trích ý, không bịa số liệu).
7. Nếu thông tin thiếu, suy luận hợp lý từ ngữ cảnh; không để null; điểm và lý do phải nhất quán.

Đoạn văn tự giới thiệu:

{user_text}
""".strip()


# ---------------------------------------------------------------------------
# Setup môi trường & Gemini
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Model 2.x thường bị 404 với API key mới. Dùng dòng Gemini 3.x.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
if API_KEY and API_KEY != "your_api_key_here":
    genai.configure(api_key=API_KEY)


@st.cache_resource
def load_ml_artifacts() -> tuple:
    """
    Load model Random Forest và 2 LabelEncoder đã train offline.
    Dùng @st.cache_resource để không load lại mỗi lần Streamlit rerun.
    """
    model_path = MODELS_DIR / "rf_model.pkl"
    field_path = MODELS_DIR / "field_encoder.pkl"
    career_path = MODELS_DIR / "career_encoder.pkl"

    missing = [p.name for p in (model_path, field_path, career_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu file model: "
            + ", ".join(missing)
            + ". Hãy chạy `python train_model.py` trước."
        )

    model = joblib.load(model_path)
    field_encoder = joblib.load(field_path)
    career_encoder = joblib.load(career_path)
    return model, field_encoder, career_encoder


def extract_json_text(raw_text: str) -> str:
    """
    Làm sạch response Gemini: bỏ markdown fence nếu vẫn bị bọc,
    hoặc cắt lấy object JSON đầu tiên tìm thấy.
    """
    text = (raw_text or "").strip()

    # Loại bỏ ```json ... ``` hoặc ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Nếu vẫn còn text thừa, lấy đoạn {...} đầu tiên
    if not text.startswith("{"):
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            text = obj_match.group(0).strip()

    return text


def call_gemini_extractor(user_text: str, list_of_all_careers: list[str]) -> dict:
    """
    Gọi Gemini lần 1: trích xuất features + reasoning + Excluded_Careers.
    list_of_all_careers = list(career_encoder.classes_).
    """
    if not API_KEY or API_KEY == "your_api_key_here":
        raise RuntimeError(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Hãy mở file .env và điền API key hợp lệ."
        )

    prompt = build_extractor_prompt(user_text, list_of_all_careers)
    llm = genai.GenerativeModel(GEMINI_MODEL)

    # max_output_tokens tăng vì Deep Reasoning (3-4 câu / kỹ năng)
    generation_config = {
        "temperature": 0.3,
        "max_output_tokens": 2048,
    }

    try:
        try:
            response = llm.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": 90},
            )
        except TypeError:
            # Một số phiên bản SDK không nhận request_options
            response = llm.generate_content(
                prompt,
                generation_config=generation_config,
            )
    except Exception as exc:
        message = str(exc)
        if "404" in message or "no longer available" in message.lower():
            raise RuntimeError(
                f"Model `{GEMINI_MODEL}` không khả dụng với API key hiện tại.\n"
                "- Đổi GEMINI_MODEL trong .env sang: gemini-3.5-flash-lite, "
                "gemini-3.1-flash-lite hoặc gemini-3.5-flash.\n"
                f"Chi tiết: {message}"
            ) from exc
        if "429" in message or "quota" in message.lower():
            raise RuntimeError(
                f"Hết quota Gemini cho model `{GEMINI_MODEL}`.\n"
                "- Đổi GEMINI_MODEL trong .env hoặc kiểm tra quota/"
                "billing trên Google AI Studio.\n"
                f"Chi tiết: {message}"
            ) from exc
        raise

    raw = getattr(response, "text", None)
    if not raw:
        raise RuntimeError(
            "Gemini không trả về nội dung text. "
            "Có thể bị chặn bởi safety filter hoặc lỗi mạng."
        )

    # Parse JSON an toàn: làm sạch markdown rồi json.loads
    cleaned = extract_json_text(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"JSON không hợp lệ sau khi làm sạch. Đoạn nhận được: {cleaned[:300]}",
            cleaned,
            exc.pos,
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("JSON trả về không phải object (dict).")

    return data


def normalize_excluded_careers(
    raw_excluded: object,
    valid_careers: list[str],
) -> list[str]:
    """
    Chuẩn hóa Excluded_Careers từ Gemini về đúng tên class trong career_encoder.
    So khớp không phân biệt hoa/thường; hỗ trợ khớp gần (vd: 'DevOps' → 'DevOps Engineer').
    """
    if raw_excluded is None:
        return []
    if not isinstance(raw_excluded, list):
        raw_excluded = [raw_excluded]

    lower_to_official = {c.strip().lower(): c for c in valid_careers}
    matched: list[str] = []
    seen: set[str] = set()

    for item in raw_excluded:
        name = str(item).strip()
        if not name:
            continue

        key = name.lower()
        official = lower_to_official.get(key)

        # Khớp gần: chuỗi Gemini nằm trong tên class hoặc ngược lại
        if official is None:
            for valid_lower, valid_name in lower_to_official.items():
                if key in valid_lower or valid_lower in key:
                    official = valid_name
                    break

        if official and official not in seen:
            matched.append(official)
            seen.add(official)

    return matched


def validate_features(data: dict, valid_careers: list[str]) -> dict:
    """
    Chuẩn hóa kiểu dữ liệu:
    - Giá trị số cho Random Forest (Field, skills, Projects, Internships)
    - Chuỗi lý do *_Reason chỉ dùng cho UI
    - Excluded_Careers: danh sách ngành cần loại khỏi Top kết quả
    """
    required_score_keys = [
        "Field",
        "Coding Skills",
        "Communication Skills",
        "Problem Solving Skills",
        "Teamwork Skills",
        "Projects",
        "Internships",
    ]
    missing = [k for k in required_score_keys if k not in data]
    if missing:
        raise KeyError(f"JSON thiếu keys: {missing}")

    skill_keys = [
        "Coding Skills",
        "Communication Skills",
        "Problem Solving Skills",
        "Teamwork Skills",
    ]

    # Phần số — đưa vào model.predict_proba()
    normalized = {
        "Field": str(data["Field"]).strip(),
        "Projects": max(0, int(data["Projects"])),
        "Internships": max(0, int(data["Internships"])),
    }

    for key in skill_keys:
        value = int(data[key])
        # Ép về thang 0-4
        normalized[key] = min(4, max(0, value))

    # Phần reasoning — chỉ phục vụ dashboard, không ảnh hưởng ML
    for skill_key, reason_key in SKILL_REASON_KEYS.items():
        reason_text = data.get(reason_key, "")
        if reason_text is None or str(reason_text).strip() == "":
            reason_text = (
                f"Gemini không cung cấp lý do chi tiết cho {skill_key}; "
                f"điểm được ghi nhận là {normalized[skill_key]}/4."
            )
        normalized[reason_key] = str(reason_text).strip()

    # Ngành bị loại trừ theo ý người dùng (phủ định / không thích)
    normalized["Excluded_Careers"] = normalize_excluded_careers(
        data.get("Excluded_Careers", []),
        valid_careers,
    )

    return normalized


def encode_field(field_value: str, field_encoder) -> int:
    """
    Encode Field bằng LabelEncoder đã lưu.
    Nếu LLM trả Field lạ (không có trong classes_), dùng giá trị mặc định = class đầu tiên.
    """
    classes = list(field_encoder.classes_)
    if field_value in classes:
        return int(field_encoder.transform([field_value])[0])

    # Thử khớp không phân biệt hoa thường
    lower_map = {c.lower(): c for c in classes}
    if field_value.lower() in lower_map:
        matched = lower_map[field_value.lower()]
        return int(field_encoder.transform([matched])[0])

    # Mặc định: class đầu tiên đã thấy lúc train
    default_field = classes[0]
    st.warning(
        f"Field '{field_value}' không có trong dữ liệu train. "
        f"Tạm dùng mặc định: '{default_field}'."
    )
    return int(field_encoder.transform([default_field])[0])


def build_feature_vector(features: dict, field_encoder) -> np.ndarray:
    """
    Tạo numpy array 1 hàng theo đúng thứ tự FEATURE_COLUMNS.
    Chỉ lấy giá trị số / Field đã encode — bỏ qua các trường *_Reason.
    """
    field_code = encode_field(features["Field"], field_encoder)

    row = [
        field_code,
        features["Coding Skills"],
        features["Communication Skills"],
        features["Problem Solving Skills"],
        features["Teamwork Skills"],
        features["Projects"],
        features["Internships"],
    ]
    return np.array([row], dtype=float)


def predict_top_careers(
    feature_vector: np.ndarray,
    model,
    career_encoder,
    excluded_careers: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Dự đoán Top-K ngành nghề bằng predict_proba, sau đó lọc Excluded_Careers.

    1) Lấy probabilities cho mọi class
    2) Ép probability = 0.0 với ngành nằm trong excluded_careers
    3) Sort giảm dần và lấy Top-K (bỏ qua ngành có % = 0)
    """
    # Lấy xác suất cho TẤT CẢ các class nghề nghiệp (hàng đầu tiên)
    # probabilities[i] tương ứng với career_encoder.classes_[i]
    probabilities = model.predict_proba(feature_vector)[0].astype(float).copy()

    # Danh sách tên ngành nghề (đã decode sẵn từ LabelEncoder)
    career_names = list(career_encoder.classes_)

    # Tập ngành bị loại — so sánh không phân biệt hoa/thường / khoảng trắng
    excluded = excluded_careers or []
    excluded_lower = {str(c).strip().lower() for c in excluded}

    # Ép xác suất về 0.0 cho các ngành bị loại trừ TRƯỚC khi sort
    for idx, career_name in enumerate(career_names):
        if str(career_name).strip().lower() in excluded_lower:
            probabilities[idx] = 0.0

    # Ghép (tên nghề, xác suất) rồi sắp xếp giảm dần theo xác suất
    paired = list(zip(career_names, probabilities))
    paired.sort(key=lambda item: item[1], reverse=True)

    # Chỉ lấy Top-K; bỏ qua ngành đã bị ép về 0
    top_results: list[dict] = []
    for career_name, prob in paired:
        if float(prob) <= 0.0:
            continue
        top_results.append(
            {
                "career": str(career_name),
                "match_percent": round(float(prob) * 100, 1),
            }
        )
        if len(top_results) >= top_k:
            break

    return top_results


# Thông báo mặc định khi Gemini lần 2 lỗi (429 / timeout / JSON hỏng)
FALLBACK_CAREER_EXPLAIN = "Không thể tải phần giải thích lúc này."


def _build_career_explain_prompt(features: dict, top_careers: list[dict]) -> str:
    """Tạo prompt cho Gemini lần 2: giải thích vì sao hồ sơ khớp từng nghề Top 5."""
    skill_profile = {
        "Field": features["Field"],
        "Coding Skills": features["Coding Skills"],
        "Communication Skills": features["Communication Skills"],
        "Problem Solving Skills": features["Problem Solving Skills"],
        "Teamwork Skills": features["Teamwork Skills"],
        "Projects": features["Projects"],
        "Internships": features["Internships"],
    }
    careers_payload = [
        {"career": item["career"], "match_percent": item["match_percent"]}
        for item in top_careers
    ]
    career_keys = [item["career"] for item in top_careers]

    return f"""
Bạn là chuyên gia hướng nghiệp.
Dựa vào hồ sơ kỹ năng của ứng viên và danh sách 5 ngành nghề phù hợp nhất (do mô hình Machine Learning xếp hạng),
hãy viết cho MỖI ngành nghề một đoạn phân tích thật chi tiết (khoảng 3-4 câu dài) giải thích tại sao hồ sơ của ứng viên
lại cực kỳ phù hợp với đặc thù của ngành đó. Hãy trích dẫn điểm mạnh của họ để thuyết phục.

Hồ sơ kỹ năng (điểm số):
{json.dumps(skill_profile, ensure_ascii=False, indent=2)}

Top 5 ngành nghề (tên tiếng Anh + % phù hợp từ model):
{json.dumps(careers_payload, ensure_ascii=False, indent=2)}

QUY TẮC BẮT BUỘC:
1. Chỉ trả về JSON thuần túy — KHÔNG markdown, KHÔNG ```json, KHÔNG giải thích ngoài JSON.
2. Các key PHẢI khớp CHÍNH XÁC tên ngành sau (tiếng Anh): {json.dumps(career_keys, ensure_ascii=False)}
3. Value của mỗi key là đoạn văn giải thích bằng tiếng Việt (3-4 câu dài).
4. Không thêm key khác ngoài 5 ngành trên.
""".strip()


def explain_top_careers_with_gemini(
    features: dict,
    top_careers: list[dict],
) -> dict[str, str]:
    """
    Gemini lần 2: giải thích độ phù hợp của hồ sơ với từng ngành trong Top 5.

    Nếu API/JSON lỗi → gán FALLBACK_CAREER_EXPLAIN cho mọi ngành (app không crash).
    Dùng GEMINI_MODEL từ .env (không hard-code model cũ dễ 404/429).
    """
    # Khởi tạo fallback cho đủ 5 nghề trước
    explanations = {
        item["career"]: FALLBACK_CAREER_EXPLAIN for item in top_careers
    }

    if not API_KEY or API_KEY == "your_api_key_here":
        return explanations

    prompt = _build_career_explain_prompt(features, top_careers)
    # Dùng cùng model đã cấu hình (ổn định hơn hard-code gemini-2.0-flash)
    llm = genai.GenerativeModel(GEMINI_MODEL)
    generation_config = {
        "temperature": 0.4,
        "max_output_tokens": 2048,
    }

    try:
        try:
            response = llm.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": 90},
            )
        except TypeError:
            response = llm.generate_content(
                prompt,
                generation_config=generation_config,
            )

        raw = getattr(response, "text", None)
        if not raw:
            return explanations

        cleaned = extract_json_text(raw)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return explanations

        # Map key không phân biệt hoa thường để tránh lệch nhẹ từ LLM
        lower_map = {str(k).strip().lower(): str(v).strip() for k, v in parsed.items()}

        for career in explanations:
            value = parsed.get(career)
            if value is None:
                value = lower_map.get(career.lower())
            if value and str(value).strip():
                explanations[career] = str(value).strip()

    except Exception:
        # 429 / timeout / JSON lỗi → giữ fallback, không làm crash app
        return explanations

    return explanations


def render_profile_dashboard(features: dict) -> None:
    """
    Dashboard minh bạch hồ sơ sau khi Gemini trích xuất:
    - 3 metric: Field / Projects / Internships
    - Expander từng kỹ năng kèm điểm + Reasoning
    """
    st.subheader("📊 Phân tích Chi tiết Hồ sơ của bạn")

    # Hàng metric tổng quan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ngành học (Field)", features["Field"])
    with col2:
        st.metric("Số lượng Dự án (Projects)", features["Projects"])
    with col3:
        st.metric("Số kỳ Thực tập (Internships)", features["Internships"])

    st.markdown("---")
    st.caption(
        "Điểm kỹ năng (0–4) bên dưới được đưa vào mô hình Random Forest. "
        "Phần lý do chỉ để giải thích, không tham gia dự đoán."
    )

    # Các khối kỹ năng + reasoning
    for skill_key, reason_key in SKILL_REASON_KEYS.items():
        score = features[skill_key]
        label = SKILL_UI_LABELS[skill_key]
        with st.expander(f"{label}: {score}/4", expanded=True):
            st.info(features[reason_key])


def render_top_careers(
    top_careers: list[dict],
    career_explanations: dict[str, str] | None = None,
    excluded_careers: list[str] | None = None,
) -> None:
    """Hiển thị Top 5 ngành nghề + thanh progress + đoạn giải thích Gemini lần 2."""
    st.subheader("🎯 Top 5 Ngành Nghề Phù Hợp Nhất")

    # Thông báo ngành đã loại trừ theo yêu cầu người dùng
    if excluded_careers:
        st.warning(
            "🚫 Hệ thống đã loại trừ các ngành nghề theo yêu cầu của bạn: "
            f"{', '.join(excluded_careers)}"
        )

    st.caption(
        "Tỷ lệ % lấy từ `predict_proba` của Random Forest (đã lọc ngành bị loại trừ). "
        "Đoạn giải thích dưới mỗi thanh do Gemini phân tích dựa trên hồ sơ kỹ năng."
    )

    if not top_careers:
        st.warning("Không có kết quả dự đoán để hiển thị.")
        return

    if career_explanations is None:
        career_explanations = {}

    # Nghề đứng đầu làm banner nhanh
    best = top_careers[0]
    st.success(
        f"Gợi ý hàng đầu: **{best['career']}** "
        f"({best['match_percent']}% phù hợp)"
    )

    for rank, item in enumerate(top_careers, start=1):
        career = item["career"]
        percent = float(item["match_percent"])
        # st.progress nhận giá trị trong [0.0, 1.0]
        progress_value = min(1.0, max(0.0, percent / 100.0))
        explanation = career_explanations.get(career, FALLBACK_CAREER_EXPLAIN)

        st.markdown(f"**#{rank} — {career}** · `{percent}%`")
        st.progress(progress_value)
        # Giải thích chi tiết vì sao hồ sơ khớp ngành này
        st.info(explanation)


# ---------------------------------------------------------------------------
# Giao diện Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Career Guidance AI",
    page_icon="🧭",
    layout="centered",
)

st.title("🧭 Career Guidance AI")
st.markdown(
    """
Hệ thống gợi ý hướng nghiệp kết hợp **Gemini LLM** (trích xuất kỹ năng + giải thích)
và **Random Forest** (dự đoán ngành nghề phù hợp).

Hãy kể về bản thân: ngành học, sở thích, kỹ năng lập trình / giao tiếp / giải quyết vấn đề,
kinh nghiệm dự án và thực tập.
"""
)

user_bio = st.text_area(
    "Đoạn văn tự giới thiệu",
    height=220,
    placeholder=(
        "Ví dụ: Em đang học ngành Công nghệ thông tin, thích lập trình Python và "
        "làm việc với dữ liệu. Em đã tham gia 3 dự án web/app và 1 kỳ thực tập backend. "
        "Em giao tiếp ổn, thích làm nhóm và giải quyết bài toán logic..."
    ),
)

analyze_clicked = st.button(
    "Phân tích và Gợi ý nghề nghiệp",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    cleaned_bio = (user_bio or "").strip()

    # Kiểm tra độ dài đầu vào
    if len(cleaned_bio) < MIN_TEXT_LENGTH:
        st.error(
            f"Văn bản quá ngắn (tối thiểu {MIN_TEXT_LENGTH} ký tự). "
            "Hãy mô tả rõ hơn về sở thích, kỹ năng và kinh nghiệm của bạn."
        )
        st.stop()

    # Load artifacts ML (giữ nguyên logic load .pkl)
    try:
        model, field_encoder, career_encoder = load_ml_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không load được model/encoder: {exc}")
        st.stop()

    # Danh sách class nghề hợp lệ — truyền vào Prompt Gemini lần 1
    list_of_all_careers = list(career_encoder.classes_)

    # Bước 1 — Gemini lần 1: skills + reasoning + Excluded_Careers
    with st.spinner("Đang phân tích hồ sơ bằng Gemini..."):
        try:
            raw_features = call_gemini_extractor(
                cleaned_bio,
                list_of_all_careers,
            )
        except json.JSONDecodeError:
            st.error(
                "Không parse được JSON từ Gemini "
                "(có thể bị lỗi format / markdown). "
                "Vui lòng thử lại hoặc diễn đạt rõ hơn."
            )
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lỗi khi gọi Gemini API: {exc}")
            st.stop()

    # Bước 2 — Validate / chuẩn hóa JSON (số cho ML + reason + excluded)
    try:
        features = validate_features(raw_features, list_of_all_careers)
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"JSON trích xuất không hợp lệ: {exc}")
        st.json(raw_features)
        st.stop()

    excluded_careers = features.get("Excluded_Careers", [])

    # Bước 3 — predict_proba → ép excluded về 0 → sort → Top 5
    try:
        vector = build_feature_vector(features, field_encoder)
        # Giữ tên cột khi predict để tránh warning của sklearn
        vector_df = pd.DataFrame(vector, columns=FEATURE_COLUMNS)

        top_careers = predict_top_careers(
            vector_df,
            model,
            career_encoder,
            excluded_careers=excluded_careers,
            top_k=5,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi dự đoán nghề nghiệp: {exc}")
        st.stop()

    # Bước 4 — Gemini lần 2: giải thích vì sao hồ sơ khớp từng nghề Top 5
    with st.spinner("AI đang phân tích độ phù hợp của bạn với từng ngành nghề..."):
        career_explanations = explain_top_careers_with_gemini(features, top_careers)

    # Bước 5 — Render UI: phân tích kỹ năng → Top 5 (có warning excluded)
    render_profile_dashboard(features)
    st.markdown("---")
    render_top_careers(
        top_careers,
        career_explanations,
        excluded_careers=excluded_careers,
    )
