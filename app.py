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
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent / "models"
REQUIRED_MODEL_FILES = (
    "rf_model.pkl",
    "field_encoder.pkl",
    "career_encoder.pkl",
    "skills_encoder.pkl",
)

st.set_page_config(
    page_title="Career Guidance AI",
    page_icon="🧭",
    layout="centered",
)

missing_model_files = [
    file_name for file_name in REQUIRED_MODEL_FILES if not (MODELS_DIR / file_name).is_file()
]
if missing_model_files:
    st.error(
        "Thiếu model artifacts trong repository: " + ", ".join(missing_model_files)
    )
    st.stop()


from model import (
    FALLBACK_CAREER_EXPLAIN,
    MAX_SKILL_SCORE,
    MIN_TEXT_LENGTH,
    SKILL_REASON_KEYS,
    SKILL_UI_LABELS,
    apply_extra_user_info,
    build_feature_vector,
    call_gemini_extractor,
    explain_top_careers_with_gemini,
    get_field_categories,
    load_ml_artifacts,
    predict_top_careers,
    validate_features,
    validate_user_input,
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


# Nạp ML Artifacts để lấy vocab dữ liệu cho UI
try:
    model, field_encoder, career_encoder, skills_encoder = load_ml_artifacts()
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


st.title("🧭 Career Student AI")
st.markdown(
    """
Khám phá hướng nghề nghiệp phù hợp với bạn  
Hệ thống sử dụng AI Gemini để phân tích kỹ năng, sở thích và trải nghiệm của bạn, sau đó kết hợp mô hình Random Forest để đề xuất những lĩnh vực nghề nghiệp phù hợp nhất.

Hãy chia sẻ về ngành học, công nghệ bạn từng dùng, dự án, thực tập, điểm mạnh, sở thích và cả những công việc bạn không muốn theo đuổi. Càng cụ thể, gợi ý càng sát với bạn.
"""
)

user_bio = st.text_area(
    "Đoạn văn tự giới thiệu",
    height=200,
    placeholder=(
        "Ví dụ: Em đang học CNTT, thích làm backend với Python và Docker, "
        "đã làm 3 dự án web và 1 kỳ thực tập. Không thích làm marketing hay kế toán..."
    ),
)

# Form thông tin bổ sung tùy chọn giúp tăng độ chính xác
with st.expander("🛠️ Cung cấp thêm thông tin chi tiết (Tùy chọn - Giúp tăng độ chính xác dự đoán)", expanded=False):
    st.caption("Nếu bài viết chưa nêu rõ, bạn có thể chọn thủ công các thông tin dưới đây để hệ thống nhận diện chính xác hơn.")
    col1, col2 = st.columns(2)
    with col1:
        field_options = ["-- Tự động nhận diện qua bài viết --"] + list_of_all_fields
        selected_field_option = st.selectbox("Chuyên ngành học (Field)", field_options)
        user_field = selected_field_option if selected_field_option != field_options[0] else None

        user_projects = st.number_input(
            "Số lượng dự án đã làm",
            min_value=-1,
            max_value=50,
            value=-1,
            help="-1 nghĩa là tự động nhận diện từ bài viết",
        )
    with col2:
        user_skills = st.multiselect("Kỹ năng / Công nghệ đã sử dụng", options=list_of_all_skills)
        user_internships = st.number_input(
            "Số kỳ thực tập",
            min_value=-1,
            max_value=20,
            value=-1,
            help="-1 nghĩa là tự động nhận diện từ bài viết",
        )

    col3, col4 = st.columns(2)
    with col3:
        user_preferred = st.multiselect("Ngành nghề mong muốn / Ưu tiên", options=list_of_all_careers)
    with col4:
        user_excluded = st.multiselect("Ngành nghề muốn loại trừ / Không thích", options=list_of_all_careers)

    enable_manual_scores = st.checkbox("Tự đánh giá thang điểm 4 kỹ năng (0–5)")
    user_manual_scores = None
    if enable_manual_scores:
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            prof_score = st.slider("💻 Kỹ năng Chuyên môn (Professional)", 0, 5, 3)
            comm_score = st.slider("🗣️ Kỹ năng Giao tiếp (Communication)", 0, 5, 3)
        with s_col2:
            prob_score = st.slider("🧩 Kỹ năng Giải quyết vấn đề (Problem Solving)", 0, 5, 3)
            team_score = st.slider("🤝 Kỹ năng Làm việc nhóm (Teamwork)", 0, 5, 3)

        user_manual_scores = {
            "Professional Skills": prof_score,
            "Communication Skills": comm_score,
            "Problem Solving Skills": prob_score,
            "Teamwork Skills": team_score,
        }

analyze_clicked = st.button(
    "Phân tích và Gợi ý nghề nghiệp",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    cleaned_bio = (user_bio or "").strip()

    # 1. Kiểm tra Anti-Spam & Ký tự đặc biệt
    is_valid, err_msg = validate_user_input(cleaned_bio)
    if not is_valid:
        st.error(f"⚠️ **Không thể phân tích**: {err_msg}")
        st.stop()

    extra_info = {
        "Field": user_field,
        "Projects": user_projects if user_projects >= 0 else None,
        "Internships": user_internships if user_internships >= 0 else None,
        "Skills": user_skills,
        "Preferred_Careers": user_preferred,
        "Excluded_Careers": user_excluded,
        "Manual_Scores": user_manual_scores,
    }

    with st.spinner(
        "Đang phân tích chuyên sâu hồ sơ bằng AI. Quá trình có thể mất khoảng một phút, vui lòng chờ..."
    ):
        try:
            raw_features = call_gemini_extractor(
                cleaned_bio,
                list_of_all_careers,
                list_of_all_fields,
                list_of_all_skills,
                extra_info=extra_info,
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
        # Ghi đè / hợp nhất thông tin bổ sung người dùng cung cấp
        features = apply_extra_user_info(features, extra_info)
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

    with st.spinner(
        "Đang tạo phần giải thích cho các nghề phù hợp nhất. Sắp hoàn tất, vui lòng chờ..."
    ):
        career_explanations = explain_top_careers_with_gemini(features, top_careers)

    render_profile_dashboard(features)
    st.markdown("---")
    render_top_careers(
        top_careers,
        career_explanations,
        preferred_careers=preferred_careers,
        excluded_careers=excluded_careers,
    )
