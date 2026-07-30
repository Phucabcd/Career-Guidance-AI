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
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn & hằng số
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "Field",
    "Coding Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]

SKILL_REASON_KEYS = {
    "Coding Skills": "Coding_Reason",
    "Communication Skills": "Communication_Reason",
    "Problem Solving Skills": "Problem_Solving_Reason",
    "Teamwork Skills": "Teamwork_Reason",
}

SKILL_UI_LABELS = {
    "Coding Skills": "💻 Kỹ năng Lập trình (Coding Skills)",
    "Communication Skills": "🗣️ Kỹ năng Giao tiếp (Communication Skills)",
    "Problem Solving Skills": "🧩 Kỹ năng Giải quyết vấn đề (Problem Solving Skills)",
    "Teamwork Skills": "🤝 Kỹ năng Làm việc nhóm (Teamwork Skills)",
}

MIN_TEXT_LENGTH = 40
FALLBACK_CAREER_EXPLAIN = "Không thể tải phần giải thích lúc này."


def build_extractor_prompt(user_text: str, list_of_all_careers: list[str]) -> str:
    """Tạo prompt Gemini lần 1 để trích xuất features và reasoning."""
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


load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
if API_KEY and API_KEY != "your_api_key_here":
    genai.configure(api_key=API_KEY)


@lru_cache(maxsize=1)
def load_ml_artifacts() -> tuple:
    """Load mô hình Random Forest và hai LabelEncoder đã train offline."""
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


def call_gemini_extractor(user_text: str, list_of_all_careers: list[str]) -> dict:
    """Gọi Gemini lần 1 để trích xuất features + reasoning + Excluded_Careers."""
    if not API_KEY or API_KEY == "your_api_key_here":
        raise RuntimeError(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Hãy mở file .env và điền API key hợp lệ."
        )

    prompt = build_extractor_prompt(user_text, list_of_all_careers)
    llm = genai.GenerativeModel(GEMINI_MODEL)

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
                "- Đổi GEMINI_MODEL trong .env hoặc kiểm tra quota/billing trên Google AI Studio.\n"
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


def normalize_excluded_careers(raw_excluded: object, valid_careers: list[str]) -> list[str]:
    """Chuẩn hóa Excluded_Careers về đúng tên class trong career_encoder."""
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
    """Chuẩn hóa kiểu dữ liệu để dùng cho mô hình và UI."""
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

    normalized = {
        "Field": str(data["Field"]).strip(),
        "Projects": max(0, int(data["Projects"])),
        "Internships": max(0, int(data["Internships"])),
    }

    for key in skill_keys:
        value = int(data[key])
        normalized[key] = min(4, max(0, value))

    for skill_key, reason_key in SKILL_REASON_KEYS.items():
        reason_text = data.get(reason_key, "")
        if reason_text is None or str(reason_text).strip() == "":
            reason_text = (
                f"Gemini không cung cấp lý do chi tiết cho {skill_key}; "
                f"điểm được ghi nhận là {normalized[skill_key]}/4."
            )
        normalized[reason_key] = str(reason_text).strip()

    normalized["Excluded_Careers"] = normalize_excluded_careers(
        data.get("Excluded_Careers", []),
        valid_careers,
    )

    return normalized


def encode_field(
    field_value: str,
    field_encoder,
    warning_callback: Callable[[str], None] | None = None,
) -> int:
    """Encode Field bằng LabelEncoder đã lưu."""
    classes = list(field_encoder.classes_)
    if field_value in classes:
        return int(field_encoder.transform([field_value])[0])

    lower_map = {c.lower(): c for c in classes}
    if field_value.lower() in lower_map:
        matched = lower_map[field_value.lower()]
        return int(field_encoder.transform([matched])[0])

    default_field = classes[0]
    if warning_callback:
        warning_callback(
            f"Field '{field_value}' không có trong dữ liệu train. "
            f"Tạm dùng mặc định: '{default_field}'."
        )
    return int(field_encoder.transform([default_field])[0])


def build_feature_vector(
    features: dict,
    field_encoder,
    warning_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Tạo numpy array 1 hàng theo đúng thứ tự FEATURE_COLUMNS."""
    field_code = encode_field(features["Field"], field_encoder, warning_callback)

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
    """Dự đoán Top-K ngành nghề và loại bỏ các ngành bị người dùng chặn."""
    probabilities = model.predict_proba(feature_vector)[0].astype(float).copy()
    career_names = list(career_encoder.classes_)

    excluded = excluded_careers or []
    excluded_lower = {str(c).strip().lower() for c in excluded}

    for idx, career_name in enumerate(career_names):
        if str(career_name).strip().lower() in excluded_lower:
            probabilities[idx] = 0.0

    paired = list(zip(career_names, probabilities))
    paired.sort(key=lambda item: item[1], reverse=True)

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
    """Gemini lần 2: giải thích độ phù hợp của hồ sơ với từng ngành Top 5."""
    explanations = {item["career"]: FALLBACK_CAREER_EXPLAIN for item in top_careers}

    if not API_KEY or API_KEY == "your_api_key_here":
        return explanations

    prompt = _build_career_explain_prompt(features, top_careers)
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
