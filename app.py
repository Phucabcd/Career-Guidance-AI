"""
app.py
Streamlit UI cho Career Guidance AI.
Logic mô hình nằm trong model.py.

Chạy:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import gdown
import streamlit as st

DRIVE_FILES = {
    'rf_model.pkl': '1hAQtY2XCos9E_Qt_ulao2tKu5vzu_1BQ',
    'field_encoder.pkl': '1zw4tL7hnGxx0P0ODIZobBGjfRveBn_tQ',
    'career_encoder.pkl': '1ky6G0V6I5PvcCFsNfxEfZ9XcC7Ar2wsc',
    'skills_encoder.pkl': '1jtgaV8gYQXpm5VnmWEZ_y5aUwXA5Z2RE',
    'model_meta.pkl': '1DhSaxA9plg5DJIC_P3XkrHdQSmtmPEY0',
}


MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Career Guidance AI",
    page_icon="🧭",
    layout="centered",
)

@st.cache_resource
def download_models():
    for file_name, file_id in DRIVE_FILES.items():
        output_path = MODELS_DIR / file_name
        if not output_path.exists():
            gdown.download(id=file_id, output=str(output_path), quiet=False)

download_models()


from model import (
    FALLBACK_CAREER_EXPLAIN,
    MAX_SKILL_SCORE,
    MIN_TEXT_LENGTH,
    SKILL_REASON_KEYS,
    SKILL_UI_LABELS,
    build_feature_vector,
    call_gemini_extractor,
    explain_top_careers_with_gemini,
    get_field_categories,
    load_ml_artifacts,
    predict_top_careers,
    validate_features,
)


def render_profile_dashboard(features: dict) -> None:
    """Hiển thị dashboard phân tích hồ sơ trên Streamlit."""
    st.subheader("Kết quả phân tích chi tiết hồ sơ của bạn")

    col1, col2, col3 = st.columns([5, 2, 2])
    with col1:
        st.metric("Chuyên ngành", features["Field"])
    with col2:
        st.metric("Số lượng Dự án", features["Projects"])
    with col3:
        st.metric("Số kỳ Thực tập", features["Internships"])

    st.markdown("---")
    st.caption(
        f"Điểm kỹ năng (0–{MAX_SKILL_SCORE}) + Skills binary được đưa vào Random Forest. "
        "Phần lý do chỉ để giải thích, không tham gia dự đoán."
    )

    for skill_key, reason_key in SKILL_REASON_KEYS.items():
        score = features[skill_key]
        label = SKILL_UI_LABELS[skill_key]
        with st.expander(f"{label}: {score}/{MAX_SKILL_SCORE}", expanded=False):
            st.info(features[reason_key])


def render_top_careers(
    top_careers: list[dict],
    career_explanations: dict[str, str] | None = None,
    preferred_careers: list[str] | None = None,
    excluded_careers: list[str] | None = None,
) -> None:
    """Hiển thị Top 5 ngành nghề với progress bar và giải thích."""
    st.subheader("🎯 Top 5 Ngành Nghề Phù Hợp Nhất")

    # if preferred_careers:
    #     st.success(
    #         "🌟 Đã tăng cường độ ưu tiên cho các ngành theo sở thích của bạn: "
    #         f"{', '.join(preferred_careers)}"
    #     )
    # if excluded_careers:
    #     st.warning(
    #         "🚫 Hệ thống đã loại trừ các ngành nghề bạn không hứng thú: "
    #         f"{', '.join(excluded_careers)}"
    #     )

    st.caption(
        "Tỷ lệ % từ Random Forest sau khi lọc (Excluded) và tăng trọng số (Preferred), "
        "đã chuẩn hóa trong Top 5. Đoạn giải thích do Gemini phân tích."
    )

    if not top_careers:
        st.warning("Không có kết quả dự đoán để hiển thị.")
        return

    if career_explanations is None:
        career_explanations = {}

    best = top_careers[0]
    st.success(
        f"Gợi ý hàng đầu: **{best['career']}** "
        f"({best['match_percent']}% phù hợp)"
    )

    for rank, item in enumerate(top_careers, start=1):
        career = item["career"]
        percent = float(item["match_percent"])
        progress_value = min(1.0, max(0.0, percent / 100.0))
        explanation = career_explanations.get(career, FALLBACK_CAREER_EXPLAIN)

        st.markdown(f"**#{rank} — {career}** · `{percent}%`")
        st.progress(progress_value)
        st.info(explanation)


st.title("🧭 Career Guidance AI")
st.markdown(
    """
Hệ thống gợi ý hướng nghiệp kết hợp **Gemini LLM** (trích xuất kỹ năng + Skills công nghệ)
và **Random Forest** (dự đoán ngành nghề phù hợp từ dataset IT mở rộng).

Hãy kể về bản thân: ngành học, sở thích, công nghệ (Python, Java, Docker,...),
dự án, thực tập. Có thể nêu rõ ngành **thích** / **không thích**.
"""
)

user_bio = st.text_area(
    "Đoạn văn tự giới thiệu",
    height=220,
    placeholder=(
        "Ví dụ: Em đang học CNTT, thích làm backend với Python và Docker, "
        "đã làm 3 dự án web và 1 kỳ thực tập. Không thích làm marketing hay kế toán..."
    ),
)

analyze_clicked = st.button(
    "Phân tích và Gợi ý nghề nghiệp",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    cleaned_bio = (user_bio or "").strip()

    if len(cleaned_bio) < MIN_TEXT_LENGTH:
        st.error(
            f"Văn bản quá ngắn (tối thiểu {MIN_TEXT_LENGTH} ký tự). "
            "Hãy mô tả rõ hơn về sở thích, kỹ năng và kinh nghiệm của bạn."
        )
        st.stop()

    try:
        model, field_encoder, career_encoder, skills_encoder = load_ml_artifacts()
        # Các thuộc tính này sẽ báo lỗi nếu file .pkl trên Drive được train theo
        # schema/loại encoder cũ, nên phải nằm trong cùng khối chẩn đoán.
        list_of_all_careers = list(career_encoder.classes_)
        list_of_all_fields = get_field_categories(field_encoder)
        list_of_all_skills = list(skills_encoder.classes_)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể nạp hoặc đọc model/encoder: {exc}")
        st.exception(exc)
        st.stop()

    with st.spinner("Đang phân tích hồ sơ..."):
        try:
            raw_features = call_gemini_extractor(
                cleaned_bio,
                list_of_all_careers,
                list_of_all_fields,
                list_of_all_skills,
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
            st.exception(exc)
            st.stop()

    try:
        features = validate_features(
            raw_features,
            list_of_all_careers,
            valid_skills=list_of_all_skills,
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"JSON trích xuất không hợp lệ: {exc}")
        st.exception(exc)
        st.json(raw_features)
        st.stop()

    preferred_careers = features.get("Preferred_Careers", [])
    excluded_careers = features.get("Excluded_Careers", [])

    try:
        vector = build_feature_vector(
            features,
            field_encoder,
            skills_encoder,
            warning_callback=st.warning,
        )
        top_careers = predict_top_careers(
            vector,
            model,
            career_encoder,
            excluded_careers=excluded_careers,
            preferred_careers=preferred_careers,
            top_k=5,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi dự đoán nghề nghiệp: {exc}")
        st.exception(exc)
        st.stop()

    with st.spinner("AI đang phân tích độ phù hợp của bạn với từng ngành nghề..."):
        career_explanations = explain_top_careers_with_gemini(features, top_careers)

    render_profile_dashboard(features)
    st.markdown("---")
    render_top_careers(
        top_careers,
        career_explanations,
        preferred_careers=preferred_careers,
        excluded_careers=excluded_careers,
    )
