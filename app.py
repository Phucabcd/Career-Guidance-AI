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
import re
import streamlit as st

PROMPT_TEST_FILE = Path(__file__).resolve().parent / "prompt_test.txt"

GENERAL_TEMPLATE: str = (
    "- Ngành học / Trường hiện tại: [Ví dụ: Sinh viên CNTT / Kinh tế / Học sinh cấp 3 / Chưa xác định...]\n"
    "- Kỹ năng & Công cụ sở trường: [Ví dụ: Excel, Python, Canva, Tiếng Anh giao tiếp, làm việc nhóm...]\n"
    "- Dự án / Trải nghiệm thực tế: [Ví dụ: Đã làm 2 đồ án môn học, tham gia câu lạc bộ, bài tập nhóm...]\n"
    "- Điểm mạnh & Sở thích: [Ví dụ: Thích làm việc với con số, thích thiết kế hình ảnh, thích giao tiếp...]\n"
    "- Mong muốn công việc: [Ví dụ: Môi trường năng động, có cơ hội học hỏi, công việc ổn định...]\n"
    "- Công việc **KHÔNG** mong muốn: [Ví dụ: Không thích làm sales, không thích công việc trực ca đêm...]"
)

DEFAULT_SAMPLE_PROMPTS: dict[str, str] = {
    "Khoa học dữ liệu (Data Science)": (
        "Em là sinh viên năm cuối ngành Khoa học dữ liệu với GPA 3.5.\n"
        "Em sử dụng thành thạo Python, Pandas, NumPy và SQL.\n"
        "Em đã thực hiện nhiều bài toán phân tích dữ liệu và xây dựng mô hình dự đoán bằng scikit-learn.\n"
        "Em thích khám phá dữ liệu, trực quan hóa bằng Power BI và Tableau.\n"
        "Em có tư duy phân tích tốt, cẩn thận và yêu thích giải quyết các bài toán thực tế từ dữ liệu."
    ),
    "AI Engineer": (
        "Em là sinh viên chuyên ngành Trí tuệ nhân tạo.\n"
        "Em có kinh nghiệm xây dựng các mô hình Deep Learning sử dụng TensorFlow và PyTorch.\n"
        "Em đã thực hiện các dự án về thị giác máy tính (Computer Vision) và xử lý ngôn ngữ tự nhiên (NLP).\n"
        "Em am hiểu về toán tối ưu, đại số tuyến tính và xác suất thống kê.\n"
        "Em mong muốn phát triển các giải pháp AI ứng dụng vào thực tế cuộc sống."
    ),
    "Backend Developer": (
        "Em xây dựng REST API bằng Java và Spring Boot, dùng PostgreSQL, Redis, Docker và viết unit test.\n"
        "Em từng thiết kế service authentication và xử lý phân quyền cho một hệ thống thương mại điện tử."
    ),
    "Business Analyst (BA)": (
        "Em là sinh viên ngành Hệ thống thông tin quản lý với GPA 3.6.\n"
        "Em có khả năng khơi gợi yêu cầu (requirement elicitation) và viết tài liệu BRD, SRS.\n"
        "Em sử dụng thành thạo các công cụ vẽ quy trình như Draw.io, Visio và công cụ quản lý dự án Jira.\n"
        "Em có tư duy logic tốt, khả năng giao tiếp giữa khách hàng và đội ngũ kỹ thuật hiệu quả."
    ),
    "Frontend Developer": (
        "Em chuyên HTML, CSS, JavaScript và React.\n"
        "Em chuyển thiết kế Figma thành giao diện responsive, tối ưu accessibility và tốc độ tải trang.\n"
        "Em đã phát triển trang quản trị có biểu đồ và form tương tác."
    ),
    "Kế toán (Accountant)": (
        "Em tốt nghiệp ngành Kế toán - Kiểm toán.\n"
        "Em nắm vững các chuẩn mực kế toán Việt Nam (VAS) và hiểu biết về IFRS.\n"
        "Em sử dụng thành thạo phần mềm kế toán MISA và Excel chuyên sâu.\n"
        "Em có kinh nghiệm trong việc hạch toán nghiệp vụ, lập báo cáo tài chính và báo cáo thuế."
    ),
}


def load_sample_prompts(file_path: Path) -> dict[str, str]:
    """Trích xuất các đoạn prompt mẫu từ prompt_test.txt."""
    prompts: dict[str, str] = {}
    if not file_path.is_file():
        return DEFAULT_SAMPLE_PROMPTS

    try:
        content = file_path.read_text(encoding="utf-8")
        blocks = re.split(r'\n(?=###|\bPrompt:|\bAC-\d+)', content)
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Kiểm tra định dạng AC-xx
            ac_match = re.search(r'AC-\d+\s*—\s*(.+?)\nPrompt:\s*(.+?)(?=\nNhãn kỳ vọng:|\Z)', block, re.DOTALL)
            if ac_match:
                title = ac_match.group(1).strip()
                body = ac_match.group(2).strip()
                body_clean = "\n".join([l.strip() for l in body.splitlines() if l.strip()])
                prompts[f"{title} (Mẫu thực tế)"] = body_clean
                continue

            # Kiểm tra định dạng Standard Prompt
            prompt_match = re.search(r'^Prompt:\s*(.+?)\n(.+)', block, re.DOTALL)
            if prompt_match:
                title = prompt_match.group(1).strip()
                raw_body = prompt_match.group(2).strip()
                lines = [line.strip() for line in raw_body.splitlines() if line.strip() and not line.strip().startswith("(Test:")]
                body_clean = "\n".join(lines)
                if body_clean and title not in prompts:
                    prompts[title] = body_clean

    except Exception:
        pass

    return prompts if prompts else DEFAULT_SAMPLE_PROMPTS


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
st.caption("Khám phá hướng nghề nghiệp phù hợp với bạn")
st.markdown(
    """
Hệ thống sẽ phân tích kỹ năng, sở thích và trải nghiệm của bạn, sau đó đề xuất những lĩnh vực nghề nghiệp phù hợp nhất.

Hãy chia sẻ về ngành học, công nghệ bạn từng dùng, dự án, thực tập, điểm mạnh, sở thích và cả những công việc bạn không muốn theo đuổi. Càng cụ thể, gợi ý càng sát với bạn.
"""
)

# Nạp danh sách mẫu tự giới thiệu từ file prompt_test.txt
sample_prompts = load_sample_prompts(PROMPT_TEST_FILE)

if "user_bio_text" not in st.session_state:
    st.session_state["user_bio_text"] = ""

st.markdown("#### Đoạn văn tự giới thiệu")

# Thanh công cụ hỗ trợ điền nhanh thiết kế theo dạng Action Bar hiện đại
col_bar1, col_bar2, col_bar3 = st.columns([3, 6, 1])

with col_bar1:
    if st.button("📝 Nạp dàn ý điền nhanh", use_container_width=True, help="Điền khung mẫu gợi ý 6 mục dành cho người chưa định hướng được ngành nghề"):
        st.session_state["user_bio_text"] = GENERAL_TEMPLATE
        st.toast("Đã nạp dàn ý điền nhanh! Hãy thay đổi nội dung trong ngoặc [...]", icon="✨")
        st.rerun()

with col_bar2:
    with st.popover("💡 Bài viết mẫu hoàn chỉnh", use_container_width=True):
        st.markdown("**Chọn bài viết mẫu hoàn chỉnh theo ngành:**")
        selected_sample_title = st.selectbox(
            "Danh sách bài mẫu (từ prompt_test.txt):",
            options=list(sample_prompts.keys()),
            key="popover_sample_select",
        )
        if selected_sample_title in sample_prompts:
            preview_text = sample_prompts[selected_sample_title]
            st.info(preview_text[:200] + ("..." if len(preview_text) > 200 else ""))
            if st.button("📋 Áp dụng bài mẫu này", use_container_width=True):
                st.session_state["user_bio_text"] = sample_prompts[selected_sample_title]
                st.toast(f"Đã nạp mẫu: {selected_sample_title}", icon="✅")
                st.rerun()

with col_bar3:
    if st.button("🗑️", use_container_width=True, help="Xóa sạch thông tin trong khung nhập"):
        st.session_state["user_bio_text"] = ""
        st.rerun()

user_bio = st.text_area(
    "Đoạn văn tự giới thiệu",
    label_visibility="collapsed",
    height=220,
    key="user_bio_text",
    placeholder=(
        "Nhấn nút '📝 Nạp dàn ý điền nhanh' ở trên để tự động chèn dàn ý gợi ý (dành cho người chưa rõ ngành nghề), "
        "hoặc tự do viết bài tự giới thiệu bản thân..."
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
            min_value=0,
            max_value=50,
            value=0,
            help="0 nghĩa là tự động nhận diện từ bài viết",
        )
    with col2:
        user_skills = st.multiselect("Kỹ năng / Công nghệ đã sử dụng", options=list_of_all_skills)
        user_internships = st.number_input(
            "Số kỳ thực tập",
            min_value=0,
            max_value=20,
            value=0,
            help="0 nghĩa là tự động nhận diện từ bài viết",
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
        "Đang phân tích chuyên sâu hồ sơ bằng AI. Quá trình có thể tốn vài giây, vui lòng chờ..."
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
