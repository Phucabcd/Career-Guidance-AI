"""
model.py
Chứa toàn bộ logic xử lý mô hình: load artifacts, gọi Gemini, validate JSON,
chuẩn hóa features, dự đoán nghề nghiệp và sinh giải thích.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import google.generativeai as genai
import joblib
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn & hằng số
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Features số — phải khớp train_model.py (NUMERIC_FEATURE_COLUMNS).
# LƯU Ý: "Field" ở đây chỉ để liệt kê logic input, KHÔNG còn là 1 cột số duy
# nhất trong vector thật — Field giờ được OneHotEncoder tách thành N cột, do
# encode_field_vector() xử lý riêng trong build_feature_vector().
FEATURE_COLUMNS = [
    "Field",
    "Professional Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]
# Alias tương thích
NUMERIC_FEATURE_COLUMNS = FEATURE_COLUMNS

SKILL_REASON_KEYS = {
    "Professional Skills": "Professional_Reason",
    "Communication Skills": "Communication_Reason",
    "Problem Solving Skills": "Problem_Solving_Reason",
    "Teamwork Skills": "Teamwork_Reason",
}

SKILL_UI_LABELS = {
    "Professional Skills": "💻 Kỹ năng Chuyên môn (Professional Skills)",
    "Communication Skills": "🗣️ Kỹ năng Giao tiếp (Communication Skills)",
    "Problem Solving Skills": "🧩 Kỹ năng Giải quyết vấn đề (Problem Solving Skills)",
    "Teamwork Skills": "🤝 Kỹ năng Làm việc nhóm (Teamwork Skills)",
}

MIN_TEXT_LENGTH = 40
FALLBACK_CAREER_EXPLAIN = "Không thể tải phần giải thích lúc này."
# Thang điểm trong dataset mới: 0–5
MAX_SKILL_SCORE = 5


def build_extractor_prompt(
    user_input: str,
    list_of_all_careers: list[str],
    list_of_all_fields: list[str],
    list_of_all_skills: list[str],
) -> str:
    """Tạo prompt Gemini lần 1: điểm số + Skills + Preferred/Excluded."""
    careers_json = json.dumps(list_of_all_careers, ensure_ascii=False)
    fields_json = json.dumps(list_of_all_fields, ensure_ascii=False)
    skills_json = json.dumps(list_of_all_skills, ensure_ascii=False)

    return f"""Bạn là bộ trích xuất đặc trưng hướng nghiệp (NLP Feature Extractor) kèm phân tích sâu (Deep Reasoning).
Nhiệm vụ: Đọc đoạn văn tự giới thiệu của người dùng, chấm điểm kỹ năng, trích xuất công nghệ/kỹ năng cụ thể (Skills),
phân tích lý do thật chi tiết, VÀ phát hiện các ngành nghề được ƯU TIÊN/YÊU THÍCH cũng như các ngành nghề bị HẠN CHẾ/KHÔNG THÍCH/TỪ CHỐI.

Danh sách ngành nghề hợp lệ (Preferred_Careers / Excluded_Careers phải dùng ĐÚNG tên này):
{careers_json}

Danh sách Field hợp lệ (chọn 1 giá trị gần nhất):
{fields_json}

Danh sách Skills/công nghệ hợp lệ trong hệ thống (chỉ chọn từ danh sách này khi điền "Skills"):
{skills_json}

QUY TẮC BẮT BUỘC:
1. Chỉ trả về JSON thuần túy — KHÔNG bọc trong markdown, KHÔNG thêm giải thích ngoài JSON, KHÔNG dùng ```json.
2. JSON phải có ĐÚNG các keys sau (không thiếu, không đổi tên):
   - "Field": string (phải thuộc danh sách Field hợp lệ; nếu không rõ hãy chọn gần nhất)
   - "Projects": integer (>= 0) — ước lượng số dự án đã làm
   - "Internships": integer (>= 0) — ước lượng số kỳ thực tập
   - "Professional Skills": integer từ 0 đến 5
   - "Professional_Reason": string (phân tích lý do điểm Professional Skills)
   - "Communication Skills": integer từ 0 đến 5
   - "Communication_Reason": string (phân tích lý do điểm Communication Skills)
   - "Problem Solving Skills": integer từ 0 đến 5
   - "Problem_Solving_Reason": string (phân tích lý do điểm Problem Solving Skills)
   - "Teamwork Skills": integer từ 0 đến 5
   - "Teamwork_Reason": string (phân tích lý do điểm Teamwork Skills)
   - "Skills": array of string — các công nghệ/kỹ năng cụ thể người dùng đề cập hoặc suy ra hợp lý; CHỈ dùng tên trong danh sách Skills hợp lệ
   - "Preferred_Careers": array of string — ngành yêu thích / ưu tiên / muốn theo đuổi
   - "Excluded_Careers": array of string — ngành không thích / chán / từ chối
3. Thang điểm kỹ năng 0-5:
   0 = không đề cập / rất yếu
   1 = rất cơ bản
   2 = cơ bản
   3 = trung bình
   4 = khá
   5 = thành thạo / nổi bật trong văn bản
4. Đối với mỗi kỹ năng điểm số, hãy viết phần phân tích lý do (Reasoning) THẬT CHI TIẾT (khoảng 3-4 câu dài). Phải phân tích sâu sắc dựa trên câu chữ của ứng viên, trích dẫn lại ý của ứng viên để chứng minh, và đưa ra nhận xét ngắn về việc kỹ năng này ảnh hưởng thế nào đến thái độ làm việc của họ.
5. Phân tích thái độ nghề nghiệp:
   - Đối chiếu các ý ưu tiên/chán ghét với danh sách nghề {careers_json}.
   - Phân loại vào "Preferred_Careers" và "Excluded_Careers". Chỉ dùng đúng tên hợp lệ. Nếu không có, trả về [].
6. "Skills": chỉ lấy từ danh sách Skills hợp lệ; nếu người dùng nhắc công nghệ gần nghĩa (vd: "học máy" → "Machine Learning"), hãy map sang tên trong danh sách. Nếu không có skill nào rõ → [].
7. Mỗi trường *_Reason phải dựa trên bằng chứng trong đoạn văn (trích ý, không bịa số liệu).
8. Nếu thông tin thiếu, suy luận hợp lý từ ngữ cảnh; không để null; điểm và lý do phải nhất quán.

Đoạn văn tự giới thiệu: "{user_input}"
""".strip()


load_dotenv(BASE_DIR / ".env")


def get_gemini_api_key() -> str:
    """Đọc API key từ .env/local environment hoặc Streamlit Cloud Secrets."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        try:
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            # Khi chạy ngoài Streamlit hoặc chưa cấu hình secrets.
            api_key = ""
    return api_key


def get_gemini_model_name() -> str:
    """Đọc model từ .env/local environment hoặc Streamlit Cloud Secrets."""
    model_name = os.getenv("GEMINI_MODEL", "").strip()
    if not model_name:
        try:
            model_name = str(st.secrets.get("GEMINI_MODEL", "")).strip()
        except Exception:
            model_name = ""
    return model_name or "gemini-3.5-flash-lite"


def ensure_gemini_configured() -> tuple[str, str]:
    """Lấy cấu hình hiện tại và cấu hình SDK khi API key hợp lệ."""
    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()
    if api_key and api_key != "your_api_key_here":
        genai.configure(api_key=api_key)
    return api_key, model_name


@lru_cache(maxsize=1)
def load_ml_artifacts() -> tuple:
    """
    Load Random Forest + field/career/skills encoders đã train offline.
    LƯU Ý: field_encoder giờ là OneHotEncoder (không còn LabelEncoder).
    """
    model_path = MODELS_DIR / "rf_model.pkl"
    field_path = MODELS_DIR / "field_encoder.pkl"
    career_path = MODELS_DIR / "career_encoder.pkl"
    skills_path = MODELS_DIR / "skills_encoder.pkl"

    missing = [
        p.name
        for p in (model_path, field_path, career_path, skills_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Thiếu file model: "
            + ", ".join(missing)
            + ". Hãy chạy `python train_model.py` trước."
        )

    model = joblib.load(model_path)
    field_encoder = joblib.load(field_path)
    career_encoder = joblib.load(career_path)
    skills_encoder = joblib.load(skills_path)
    model.n_jobs = 1
    return model, field_encoder, career_encoder, skills_encoder


def extract_json_text(raw_text: str) -> str:
    """Làm sạch response Gemini: bỏ markdown fence hoặc cắt lấy object JSON đầu tiên."""
    text = (raw_text or "").strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            text = obj_match.group(0).strip()

    return text


def call_gemini_extractor(
    user_text: str,
    list_of_all_careers: list[str],
    list_of_all_fields: list[str],
    list_of_all_skills: list[str],
) -> dict:
    """Gọi Gemini lần 1: điểm số + Skills + Preferred/Excluded."""
    api_key, gemini_model = ensure_gemini_configured()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Hãy cài đặt GEMINI_API_KEY trong file .env hoặc Streamlit Cloud Secrets."
        )

    prompt = build_extractor_prompt(
        user_text,
        list_of_all_careers,
        list_of_all_fields,
        list_of_all_skills,
    )
    llm = genai.GenerativeModel(gemini_model)

    generation_config = {
        "temperature": 0.3,
        "max_output_tokens": 3072,
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
    except Exception as exc:
        message = str(exc)
        if "404" in message or "no longer available" in message.lower():
            raise RuntimeError(
                f"Model `{gemini_model}` không khả dụng với API key hiện tại.\n"
                "- Đổi GEMINI_MODEL trong .env hoặc Streamlit Cloud Secrets sang: gemini-3.5-flash-lite, "
                "gemini-3.1-flash-lite hoặc gemini-3.5-flash.\n"
                f"Chi tiết: {message}"
            ) from exc
        if "429" in message or "quota" in message.lower():
            raise RuntimeError(
                f"Hết quota Gemini cho model `{gemini_model}`.\n"
                "- Đổi GEMINI_MODEL trong .env/Streamlit Cloud Secrets hoặc kiểm tra quota/billing trên Google AI Studio.\n"
                f"Chi tiết: {message}"
            ) from exc
        raise

    raw = getattr(response, "text", None)
    if not raw:
        raise RuntimeError(
            "Gemini không trả về nội dung text. "
            "Có thể bị chặn bởi safety filter hoặc lỗi mạng."
        )

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


def normalize_career_list(raw_list: object, valid_items: list[str]) -> list[str]:
    """
    Chuẩn hóa list tên (Career / Skills / Field-like) về đúng vocabulary.
    So khớp .strip().lower(); hỗ trợ khớp gần.
    """
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    lower_to_official = {c.strip().lower(): c for c in valid_items}
    matched: list[str] = []
    seen: set[str] = set()

    for item in raw_list:
        name = str(item).strip()
        if not name:
            continue

        key = name.lower()
        official = lower_to_official.get(key)

        if official is None:
            for valid_lower, valid_name in lower_to_official.items():
                if key in valid_lower or valid_lower in key:
                    official = valid_name
                    break

        if official and official not in seen:
            matched.append(official)
            seen.add(official)

    return matched


def normalize_excluded_careers(raw_excluded: object, valid_careers: list[str]) -> list[str]:
    """Alias tương thích."""
    return normalize_career_list(raw_excluded, valid_careers)


def validate_features(
    data: dict,
    valid_careers: list[str],
    valid_skills: list[str] | None = None,
) -> dict:
    """Chuẩn hóa kiểu dữ liệu để dùng cho mô hình và UI."""
    required_score_keys = [
        "Field",
        "Professional Skills",
        "Communication Skills",
        "Problem Solving Skills",
        "Teamwork Skills",
        "Projects",
        "Internships",
    ]
    missing = [k for k in required_score_keys if k not in data]
    if missing:
        raise KeyError(f"JSON thiếu keys: {missing}")

    score_keys = [
        "Professional Skills",
        "Communication Skills",
        "Problem Solving Skills",
        "Teamwork Skills",
    ]

    normalized = {
        "Field": str(data["Field"]).strip(),
        "Projects": max(0, int(data["Projects"])),
        "Internships": max(0, int(data["Internships"])),
    }

    for key in score_keys:
        value = int(data[key])
        normalized[key] = min(MAX_SKILL_SCORE, max(0, value))

    for skill_key, reason_key in SKILL_REASON_KEYS.items():
        reason_text = data.get(reason_key, "")
        if reason_text is None or str(reason_text).strip() == "":
            reason_text = (
                f"Gemini không cung cấp lý do chi tiết cho {skill_key}; "
                f"điểm được ghi nhận là {normalized[skill_key]}/{MAX_SKILL_SCORE}."
            )
        normalized[reason_key] = str(reason_text).strip()

    # Skills công nghệ (cột Skills trong CSV)
    vocabulary = valid_skills or []
    normalized["Skills"] = normalize_career_list(data.get("Skills", []), vocabulary)

    preferred = normalize_career_list(data.get("Preferred_Careers", []), valid_careers)
    excluded = normalize_career_list(data.get("Excluded_Careers", []), valid_careers)

    # Nếu một ngành vừa preferred vừa excluded → ưu tiên loại trừ
    excluded_lower = {c.lower() for c in excluded}
    preferred = [c for c in preferred if c.lower() not in excluded_lower]

    normalized["Preferred_Careers"] = preferred
    normalized["Excluded_Careers"] = excluded

    return normalized


def get_field_categories(field_encoder) -> list[str]:
    """
    Lấy danh sách toàn bộ giá trị Field mà encoder đã học được lúc train.
    Tách thành hàm riêng để app.py không cần biết field_encoder là loại
    encoder nào (OneHotEncoder dùng .categories_[0], LabelEncoder cũ dùng
    .classes_) — tránh lặp lại lỗi AttributeError khi đổi loại encoder.
    """
    return list(field_encoder.categories_[0])


def encode_field_vector(
    field_value: str,
    field_encoder,
    warning_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Encode Field bằng OneHotEncoder đã lưu (kèm khớp gần trước khi one-hot).

    KHÁC với encode_field() cũ (LabelEncoder trả về 1 số nguyên): hàm này trả
    về CẢ MỘT VECTOR (nhiều phần tử 0/1), vì lúc train Field đã được one-hot
    thành nhiều cột riêng.

    Field lạ hoàn toàn (không khớp gần được với category nào đã train) sẽ để
    OneHotEncoder tự xử lý (đã fit với handle_unknown="ignore") → trả về
    vector toàn số 0, tức model dự đoán mà KHÔNG có thông tin Field cho hồ sơ
    này, thay vì ép về 1 field mặc định như cách làm cũ (an toàn hơn, tránh
    suy diễn sai lệch).
    """
    categories = list(field_encoder.categories_[0])
    value = str(field_value).strip()

    matched = value if value in categories else None

    if matched is None:
        lower_map = {c.lower(): c for c in categories}
        matched = lower_map.get(value.lower())

    if matched is None:
        for c in categories:
            if value.lower() in c.lower() or c.lower() in value.lower():
                matched = c
                break

    if matched is None:
        if warning_callback:
            warning_callback(
                f"Field '{field_value}' không có trong dữ liệu train. "
                "Model sẽ dự đoán mà không dùng thông tin Field cho hồ sơ này."
            )
        matched = value  # OneHotEncoder tự xử lý (handle_unknown="ignore" -> vector 0)

    vector = field_encoder.transform([[matched]])[0]
    return np.asarray(vector, dtype=float)


def build_feature_vector(
    features: dict,
    field_encoder,
    skills_encoder,
    warning_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Tạo numpy array 1 hàng, ĐÚNG thứ tự đã dùng khi train (train_model.py):
    [Field one-hot | Professional, Communication, Problem Solving, Teamwork,
     Projects, Internships | skill binary vector]
    """
    field_vector = encode_field_vector(features["Field"], field_encoder, warning_callback)

    numeric_row = np.array(
        [
            features["Professional Skills"],
            features["Communication Skills"],
            features["Problem Solving Skills"],
            features["Teamwork Skills"],
            features["Projects"],
            features["Internships"],
        ],
        dtype=float,
    )

    skill_list = features.get("Skills", []) or []
    # MultiLabelBinarizer.transform cần list-of-lists
    skill_vector = skills_encoder.transform([skill_list]).astype(float)[0]

    return np.hstack([field_vector, numeric_row, skill_vector]).reshape(1, -1)


# Hệ số tăng trọng số cho ngành được ưu tiên (Preferred_Careers)
PREFERRED_BOOST_MULTIPLIER = 1.5
PREFERRED_BOOST_ADD = 0.2


def predict_top_careers(
    feature_vector: np.ndarray,
    model,
    career_encoder,
    excluded_careers: list[str] | None = None,
    preferred_careers: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Dự đoán Top-K bằng predict_proba, sau đó:
    - Excluded_Careers → prob = 0.0
    - Preferred_Careers → boost (nhân + cộng, clamp tối đa 1.0)
    - Sort giảm dần, lấy Top-K (bỏ qua prob = 0)
    - Chuẩn hóa lại % trên Top-K về tổng ~100% để hiển thị
    """
    probabilities = model.predict_proba(feature_vector)[0].astype(float).copy()
    career_names = list(career_encoder.classes_)

    # results = [{"career": class_name, "prob": prob}, ...]
    results = [
        {"career": str(name), "prob": float(prob)}
        for name, prob in zip(career_names, probabilities)
    ]

    excluded_lower = {str(c).strip().lower() for c in (excluded_careers or [])}
    preferred_lower = {str(c).strip().lower() for c in (preferred_careers or [])}

    for item in results:
        career_key = item["career"].strip().lower()

        # 1) Loại trừ trước (ưu tiên hơn boost)
        if career_key in excluded_lower:
            item["prob"] = 0.0
            continue

        # 2) Tăng trọng số ngành được ưu tiên
        if career_key in preferred_lower:
            boosted = item["prob"] * PREFERRED_BOOST_MULTIPLIER + PREFERRED_BOOST_ADD
            item["prob"] = min(1.0, boosted)

    # Sort theo prob giảm dần
    results.sort(key=lambda x: x["prob"], reverse=True)

    # Lấy Top-K, bỏ ngành prob = 0
    top_raw = [r for r in results if r["prob"] > 0.0][:top_k]

    if not top_raw:
        return []

    # Chuẩn hóa % hiển thị về tổng 100% trong Top-K
    total_prob = sum(r["prob"] for r in top_raw)
    top_results: list[dict] = []
    for item in top_raw:
        if total_prob > 0:
            match_percent = round((item["prob"] / total_prob) * 100, 1)
        else:
            match_percent = 0.0
        top_results.append(
            {
                "career": item["career"],
                "match_percent": match_percent,
            }
        )

    return top_results


def _build_career_explain_prompt(features: dict, top_careers: list[dict]) -> str:
    """Tạo prompt cho Gemini lần 2: giải thích vì sao hồ sơ khớp từng nghề Top 5."""
    skill_profile = {
        "Field": features["Field"],
        "Professional Skills": features["Professional Skills"],
        "Communication Skills": features["Communication Skills"],
        "Problem Solving Skills": features["Problem Solving Skills"],
        "Teamwork Skills": features["Teamwork Skills"],
        "Projects": features["Projects"],
        "Internships": features["Internships"],
        "Skills": features.get("Skills", []),
    }
    careers_payload = [
        {"career": item["career"], "match_percent": item["match_percent"]}
        for item in top_careers
    ]
    career_keys = [item["career"] for item in top_careers]

    return f"""
Bạn là chuyên gia hướng nghiệp.
Dựa vào hồ sơ kỹ năng của ứng viên và danh sách ngành nghề phù hợp nhất,
hãy viết mỗi ngành nghề 1 đoạn ngắn, tự nhiên, bằng tiếng Việt, khoảng 1-2 câu.
Nội dung phải:
- thân thiện, dễ hiểu
- nêu rõ lý do chính khiến ứng viên phù hợp
- không quá giống máy, không dùng quá nhiều cấu trúc câu văn bản cứng nhắc

Hồ sơ kỹ năng (điểm số + Skills):
{json.dumps(skill_profile, ensure_ascii=False, indent=2)}

Top ngành nghề (tên tiếng Anh + % phù hợp từ model):
{json.dumps(careers_payload, ensure_ascii=False, indent=2)}

QUY TẮC BẮT BUỘC:
1. Chỉ trả về JSON thuần túy — KHÔNG markdown, KHÔNG ```json, KHÔNG giải thích ngoài JSON.
2. Các key PHẢI khớp CHÍNH XÁC tên ngành sau (tiếng Anh): {json.dumps(career_keys, ensure_ascii=False)}
3. Value của mỗi key là đoạn văn giải thích bằng tiếng Việt (3-4 câu dài).
4. Không thêm key khác ngoài các ngành trên.
""".strip()


def explain_top_careers_with_gemini(
    features: dict,
    top_careers: list[dict],
) -> dict[str, str]:
    """Gemini lần 2: giải thích độ phù hợp của hồ sơ với từng ngành Top 5."""
    explanations = {item["career"]: FALLBACK_CAREER_EXPLAIN for item in top_careers}

    api_key, gemini_model = ensure_gemini_configured()
    if not api_key or api_key == "your_api_key_here":
        return explanations

    prompt = _build_career_explain_prompt(features, top_careers)
    llm = genai.GenerativeModel(gemini_model)
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

        lower_map = {str(k).strip().lower(): str(v).strip() for k, v in parsed.items()}

        for career in explanations:
            value = parsed.get(career)
            if value is None:
                value = lower_map.get(career.lower())
            if value and str(value).strip():
                explanations[career] = str(value).strip()

    except Exception:
        return explanations

    return explanations
