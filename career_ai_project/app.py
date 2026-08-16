"""
app.py
Streamlit UI cho Career Guidance AI.
Logic mô hình nằm trong model.py.

Chạy:
    streamlit run app.py
"""

from __future__ import annotations

import json
import html

import streamlit as st

from model import (
    FALLBACK_CAREER_EXPLAIN,
    MAX_SKILL_SCORE,
    MIN_TEXT_LENGTH,
    SKILL_REASON_KEYS,
    SKILL_UI_LABELS,
    build_feature_vector,
    call_gemini_extractor,
    explain_top_careers_with_gemini,
    get_field_classes,
    load_ml_artifacts,
    predict_top_careers,
    validate_features,
)

# ---------------------------------------------------------------------------
# Palette — xanh brand #1B3FD6
# ---------------------------------------------------------------------------
# #1B3FD6  500   — primary
# #1634B5  600   — primary đậm / CTA
# #122A94  700   — hover / nhấn
# #4A63DB  400   — accent sáng
# #8FA0E8  300   — viền nhẹ
# #E0E6FA  100   — nền card
# #EEF1FC  50    — nền trang
# #0A1748  900   — chữ đậm
# #0E2070  800   — chữ phụ

SKY = {
    "50": "#EEF1FC",
    "100": "#E0E6FA",
    "200": "#C5CEF5",
    "300": "#8FA0E8",
    "400": "#4A63DB",
    "500": "#1B3FD6",
    "600": "#1634B5",
    "700": "#122A94",
    "800": "#0E2070",
    "900": "#0A1748",
}


def inject_styles() -> None:
    """CSS theme xanh brand #1B3FD6 — giao diện ứng dụng hiện đại."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Outfit:wght@500;600;700&display=swap');

html, body, [class*="css"] {{
  font-family: "DM Sans", system-ui, sans-serif;
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, {SKY["200"]}66 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, {SKY["300"]}44 0%, transparent 50%),
    linear-gradient(180deg, {SKY["50"]} 0%, #FFFFFF 42%, {SKY["50"]} 100%);
  color: {SKY["900"]};
}}

/* Ẩn chrome Streamlit mặc định cho cảm giác app */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.block-container {{
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 880px !important;
}}

/* Hero */
.cg-hero {{
  background: linear-gradient(135deg, {SKY["600"]} 0%, {SKY["500"]} 48%, {SKY["400"]} 100%);
  border-radius: 24px;
  padding: 2rem 2rem 1.75rem;
  color: #fff;
  box-shadow: 0 20px 50px -20px {SKY["600"]}99;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}}
.cg-hero::after {{
  content: "";
  position: absolute;
  right: -40px;
  top: -40px;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: rgba(255,255,255,0.12);
}}
.cg-hero::before {{
  content: "";
  position: absolute;
  right: 60px;
  bottom: -50px;
  width: 140px;
  height: 140px;
  border-radius: 50%;
  background: rgba(255,255,255,0.08);
}}
.cg-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.28);
  backdrop-filter: blur(8px);
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  margin-bottom: 0.85rem;
}}
.cg-hero h1 {{
  font-family: "Outfit", "DM Sans", sans-serif;
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  line-height: 1.2;
  position: relative;
  z-index: 1;
}}
.cg-hero p {{
  margin: 0;
  opacity: 0.95;
  font-size: 1.02rem;
  line-height: 1.55;
  max-width: 36rem;
  position: relative;
  z-index: 1;
}}

/* Cards */
.cg-card {{
  background: #FFFFFF;
  border: 1px solid {SKY["200"]};
  border-radius: 18px;
  padding: 1.25rem 1.35rem;
  box-shadow: 0 8px 24px -16px {SKY["700"]}33;
  margin-bottom: 1rem;
}}
.cg-section-title {{
  font-family: "Outfit", "DM Sans", sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: {SKY["900"]};
  margin: 0 0 0.35rem 0;
}}
.cg-section-sub {{
  color: {SKY["800"]};
  font-size: 0.92rem;
  margin: 0 0 1rem 0;
  opacity: 0.85;
}}

/* Metric tiles */
.cg-metric-grid {{
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr;
  gap: 0.75rem;
  margin: 0.75rem 0 1rem 0;
}}
@media (max-width: 700px) {{
  .cg-metric-grid {{ grid-template-columns: 1fr; }}
}}
.cg-metric {{
  background: linear-gradient(180deg, {SKY["50"]} 0%, #FFFFFF 100%);
  border: 1px solid {SKY["200"]};
  border-radius: 14px;
  padding: 1rem 1.1rem;
}}
.cg-metric .label {{
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: {SKY["700"]};
  margin-bottom: 0.35rem;
}}
.cg-metric .value {{
  font-family: "Outfit", "DM Sans", sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: {SKY["900"]};
  line-height: 1.25;
  word-break: break-word;
}}

/* Top-3 medals */
.cg-top3 {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin: 1rem 0 1.25rem 0;
}}
@media (max-width: 700px) {{
  .cg-top3 {{ grid-template-columns: 1fr; }}
}}
.cg-medal {{
  background: linear-gradient(160deg, {SKY["100"]} 0%, #FFFFFF 70%);
  border: 1px solid {SKY["200"]};
  border-radius: 16px;
  padding: 1.1rem 1rem;
  text-align: center;
  position: relative;
}}
.cg-medal.rank-1 {{
  background: linear-gradient(160deg, {SKY["500"]} 0%, {SKY["600"]} 100%);
  border-color: {SKY["600"]};
  color: #fff;
  box-shadow: 0 12px 28px -12px {SKY["600"]}aa;
}}
.cg-medal.rank-1 .rank,
.cg-medal.rank-1 .pct,
.cg-medal.rank-1 .name {{ color: #fff; }}
.cg-medal .rank {{
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {SKY["700"]};
  margin-bottom: 0.4rem;
}}
.cg-medal .pct {{
  font-family: "Outfit", "DM Sans", sans-serif;
  font-size: 1.85rem;
  font-weight: 700;
  color: {SKY["600"]};
  line-height: 1;
  margin-bottom: 0.45rem;
}}
.cg-medal .name {{
  font-size: 0.92rem;
  font-weight: 600;
  color: {SKY["900"]};
  line-height: 1.35;
}}

/* Career row cards */
.cg-career {{
  background: #FFFFFF;
  border: 1px solid {SKY["200"]};
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  margin-bottom: 0.85rem;
  box-shadow: 0 6px 18px -14px {SKY["800"]}44;
}}
.cg-career-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.75rem;
  margin-bottom: 0.55rem;
}}
.cg-career-head .title {{
  font-family: "Outfit", "DM Sans", sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: {SKY["900"]};
}}
.cg-career-head .pct {{
  font-weight: 700;
  color: {SKY["600"]};
  white-space: nowrap;
  background: {SKY["100"]};
  border: 1px solid {SKY["200"]};
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.85rem;
}}
.cg-bar {{
  height: 8px;
  background: {SKY["100"]};
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 0.75rem;
}}
.cg-bar > span {{
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, {SKY["400"]}, {SKY["600"]});
}}
.cg-explain {{
  background: {SKY["50"]};
  border-left: 3px solid {SKY["400"]};
  border-radius: 0 10px 10px 0;
  padding: 0.75rem 0.9rem;
  color: {SKY["800"]};
  font-size: 0.95rem;
  line-height: 1.55;
}}

/* Chips */
.cg-chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.5rem 0 0.75rem 0;
}}
.cg-chip {{
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
}}
.cg-chip.ok {{
  background: {SKY["100"]};
  color: {SKY["800"]};
  border: 1px solid {SKY["300"]};
}}
.cg-chip.no {{
  background: #FEF2F2;
  color: #991B1B;
  border: 1px solid #FECACA;
}}

/* Skill score pill in expander label area via caption */
.cg-caption {{
  color: {SKY["800"]};
  font-size: 0.9rem;
  opacity: 0.9;
  margin: 0.25rem 0 0.85rem 0;
}}

.cg-run-status {{
  margin: 0.65rem 0 0.25rem 0;
  padding: 0.7rem 0.95rem;
  border-radius: 12px;
  border: 1px solid {SKY["200"]};
  background: {SKY["50"]};
  color: {SKY["700"]};
  font-weight: 600;
  font-size: 0.95rem;
}}
.cg-run-status.done {{
  border-color: #BBF7D0;
  background: #F0FDF4;
  color: #166534;
}}

/* Inputs — 1 border duy nhất (tránh double border BaseWeb + textarea) */
div[data-testid="stTextArea"] [data-baseweb="base-input"],
div[data-testid="stTextArea"] [data-baseweb="textarea"],
div[data-testid="stTextArea"] > div > div {{
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}}
div[data-testid="stTextArea"] textarea {{
  border-radius: 14px !important;
  border: 1.5px solid {SKY["200"]} !important;
  outline: none !important;
  background: #FFFFFF !important;
  color: {SKY["900"]} !important;
  font-size: 1rem !important;
  min-height: 180px !important;
  box-shadow: 0 4px 16px -12px {SKY["700"]}55 !important;
}}
div[data-testid="stTextArea"] textarea:focus {{
  border-color: {SKY["500"]} !important;
  outline: none !important;
  box-shadow: 0 0 0 3px {SKY["200"]} !important;
}}

/* Select / multiselect — cùng style (nền trắng + viền brand) */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
  border-radius: 12px !important;
  border: 1.5px solid {SKY["200"]} !important;
  background-color: #FFFFFF !important;
  min-height: 42px !important;
  box-shadow: 0 4px 16px -12px {SKY["700"]}55 !important;
}}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {{
  border-color: {SKY["300"]} !important;
}}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {{
  border-color: {SKY["500"]} !important;
  box-shadow: 0 0 0 3px {SKY["200"]} !important;
}}
div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] input {{
  background: transparent !important;
  color: {SKY["900"]} !important;
}}
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
  background-color: {SKY["100"]} !important;
  border: 1px solid {SKY["200"]} !important;
  color: {SKY["800"]} !important;
  border-radius: 8px !important;
}}

/* Primary button */
div[data-testid="stButton"] > button {{
  background: linear-gradient(135deg, {SKY["500"]} 0%, {SKY["600"]} 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
  font-size: 1.02rem !important;
  padding: 0.7rem 1.2rem !important;
  box-shadow: 0 10px 24px -12px {SKY["600"]}cc !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}
div[data-testid="stButton"] > button:hover {{
  background: linear-gradient(135deg, {SKY["600"]} 0%, {SKY["700"]} 100%) !important;
  transform: translateY(-1px);
  box-shadow: 0 14px 28px -12px {SKY["700"]}cc !important;
}}
div[data-testid="stButton"] > button:focus {{
  box-shadow: 0 0 0 3px {SKY["200"]}, 0 10px 24px -12px {SKY["600"]}cc !important;
}}

/* Expanders */
div[data-testid="stExpander"] {{
  background: #FFFFFF;
  border: 1px solid {SKY["200"]} !important;
  border-radius: 14px !important;
  margin-bottom: 0.55rem;
  box-shadow: 0 4px 14px -12px {SKY["800"]}33;
}}
div[data-testid="stExpander"] details summary {{
  font-weight: 600;
  color: {SKY["900"]};
}}

/* Alerts restyle lightly */
div[data-testid="stAlert"] {{
  border-radius: 12px !important;
}}

/* Progress (fallback if still used) */
div[data-testid="stProgress"] > div > div > div > div {{
  background-color: {SKY["500"]} !important;
}}

label[data-testid="stWidgetLabel"] p {{
  font-weight: 600 !important;
  color: {SKY["500"]} !important;
}}

hr {{
  border: none;
  border-top: 1px solid {SKY["200"]};
  margin: 1.25rem 0;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="cg-hero">
  <div class="cg-badge">Gemini LLM · Random Forest · Multi-industry</div>
  <h1>Career Guidance AI</h1>
  <p>
    Hệ thống gợi ý hướng nghiệp kết hợp <strong>Gemini LLM</strong>
    (trích xuất kỹ năng chuyên môn + kỹ năng mềm) và <strong>Random Forest</strong>
    (dự đoán ngành nghề phù hợp từ dataset đa ngành).
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_dashboard(features: dict) -> None:
    """Hiển thị dashboard phân tích hồ sơ trên Streamlit."""
    field = html.escape(str(features["Field"]))
    projects = html.escape(str(features["Projects"]))
    internships = html.escape(str(features["Internships"]))

    st.markdown(
        f"""
<div class="cg-card">
  <h2 class="cg-section-title">Kết quả phân tích chi tiết hồ sơ của bạn</h2>
  <div class="cg-metric-grid">
    <div class="cg-metric">
      <div class="label">Chuyên ngành</div>
      <div class="value">{field}</div>
    </div>
    <div class="cg-metric">
      <div class="label">Số lượng Dự án</div>
      <div class="value">{projects}</div>
    </div>
    <div class="cg-metric">
      <div class="label">Số kỳ Thực tập</div>
      <div class="value">{internships}</div>
    </div>
  </div>
  <p class="cg-caption">
    Điểm kỹ năng (0–{MAX_SKILL_SCORE}) + Skills binary được đưa vào Random Forest.
    Phần lý do chỉ để giải thích, không tham gia dự đoán.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    skills = features.get("Skills") or []
    if skills:
        chips = "".join(
            f'<span class="cg-chip ok">{html.escape(str(s))}</span>' for s in skills
        )
        st.markdown(
            f"""
<div class="cg-card">
  <h3 class="cg-section-title" style="font-size:1.05rem">Skills đã trích xuất</h3>
  <div class="cg-chip-row">{chips}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="cg-section-title" style="font-size:1.1rem;margin:0.5rem 0 0.65rem">Điểm kỹ năng mềm</p>',
        unsafe_allow_html=True,
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
    """Hiển thị Top-K ngành nghề với nhấn mạnh Top-3."""
    st.markdown(
        """
<div class="cg-card">
  <h2 class="cg-section-title">Top ngành nghề phù hợp</h2>
  <p class="cg-section-sub">
    Hệ thống tối ưu theo <strong>Top-3 / Top-5</strong> (phù hợp bài toán gợi ý đa ngành).
    % bên dưới đã chuẩn hóa trong danh sách gợi ý sau khi áp Preferred/Excluded.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if preferred_careers:
        chips = "".join(
            f'<span class="cg-chip ok">{html.escape(c)}</span>' for c in preferred_careers
        )
        st.markdown(
            f'<div class="cg-card"><strong style="color:{SKY["700"]}">Ưu tiên</strong>'
            f'<div class="cg-chip-row">{chips}</div></div>',
            unsafe_allow_html=True,
        )
    if excluded_careers:
        chips = "".join(
            f'<span class="cg-chip no">{html.escape(c)}</span>' for c in excluded_careers
        )
        st.markdown(
            f'<div class="cg-card"><strong style="color:#991B1B">Đã loại</strong>'
            f'<div class="cg-chip-row">{chips}</div></div>',
            unsafe_allow_html=True,
        )

    if not top_careers:
        st.warning("Không có kết quả dự đoán để hiển thị.")
        return

    if career_explanations is None:
        career_explanations = {}

    # Top-3 nổi bật
    top3 = top_careers[:3]
    medals_html = ['<div class="cg-top3">']
    for i, item in enumerate(top3):
        rank_class = f" rank-{i + 1}" if i == 0 else ""
        medals_html.append(
            f"""
<div class="cg-medal{rank_class}">
  <div class="rank">#{i + 1}</div>
  <div class="pct">{html.escape(str(item["match_percent"]))}%</div>
  <div class="name">{html.escape(str(item["career"]))}</div>
</div>
            """
        )
    medals_html.append("</div>")
    st.markdown("".join(medals_html), unsafe_allow_html=True)

    for rank, item in enumerate(top_careers, start=1):
        career = item["career"]
        percent = float(item["match_percent"])
        progress_pct = min(100.0, max(0.0, percent))
        explanation = career_explanations.get(career, FALLBACK_CAREER_EXPLAIN)
        label = "Top-3" if rank <= 3 else f"#{rank}"
        st.markdown(
            f"""
<div class="cg-career">
  <div class="cg-career-head">
    <div class="title">{html.escape(label)} — {html.escape(str(career))}</div>
    <div class="pct">{html.escape(str(percent))}%</div>
  </div>
  <div class="cg-bar"><span style="width:{progress_pct}%"></span></div>
  <div class="cg-explain">{html.escape(str(explanation))}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
FIELD_AUTO = "— Tự động từ đoạn giới thiệu —"

st.set_page_config(
    page_title="Career Guidance AI",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_styles()
render_hero()

# Load vocab sớm để đổ vào select (cần đã train model)
_load_error: str | None = None
try:
    model, field_encoder, career_encoder, skills_encoder, extras = load_ml_artifacts()
    list_of_all_careers = sorted(str(c) for c in career_encoder.classes_)
    list_of_all_fields = sorted(get_field_classes(field_encoder))
    list_of_all_skills = sorted(str(s) for s in skills_encoder.classes_)
except FileNotFoundError as exc:
    _load_error = str(exc)
    model = field_encoder = career_encoder = skills_encoder = extras = None
    list_of_all_careers, list_of_all_fields, list_of_all_skills = [], [], []
except Exception as exc:  # noqa: BLE001
    _load_error = f"Không load được model/encoder: {exc}"
    model = field_encoder = career_encoder = skills_encoder = extras = None
    list_of_all_careers, list_of_all_fields, list_of_all_skills = [], [], []

if _load_error:
    st.error(_load_error)

st.markdown(
    '<p class="cg-section-title" style="margin:1rem 0 0.35rem 0">Thông tin hồ sơ</p>',
    unsafe_allow_html=True,
)

user_bio = st.text_area(
    "Đoạn văn tự giới thiệu",
    height=180,
    placeholder=(
        "Ví dụ: Em đang học Điều dưỡng, thích chăm sóc bệnh nhân, "
        "đã thực tập 2 kỳ tại bệnh viện. Không thích làm việc văn phòng thuần túy..."
    ),
)

col_a, col_b = st.columns(2)
with col_a:
    selected_field = st.selectbox(
        "Lĩnh vực (Field)",
        options=[FIELD_AUTO] + list_of_all_fields,
        index=0,
        help="Để trống (tự động) nếu muốn AI suy luận từ đoạn giới thiệu.",
        disabled=bool(_load_error),
    )
with col_b:
    c1, c2 = st.columns(2)
    with c1:
        user_projects = st.selectbox(
            "Số dự án",
            options=list(range(0, 21)),
            index=0,
            help="0 = để AI ước lượng từ bio.",
            disabled=bool(_load_error),
        )
    with c2:
        user_internships = st.selectbox(
            "Số kỳ thực tập",
            options=list(range(0, 11)),
            index=0,
            help="0 = để AI ước lượng từ bio.",
            disabled=bool(_load_error),
        )

selected_skills = st.multiselect(
    "Kỹ năng / công nghệ bạn có",
    options=list_of_all_skills,
    default=[],
    help="Chọn từ danh sách hệ thống. AI vẫn có thể bổ sung thêm từ đoạn giới thiệu.",
    disabled=bool(_load_error),
)

col_p, col_e = st.columns(2)
with col_p:
    selected_preferred = st.multiselect(
        "Nghề quan tâm / ưu tiên",
        options=list_of_all_careers,
        default=[],
        help="Các nghề bạn muốn được ưu tiên trong gợi ý.",
        disabled=bool(_load_error),
    )
with col_e:
    selected_excluded = st.multiselect(
        "Nghề không muốn",
        options=list_of_all_careers,
        default=[],
        help="Các nghề sẽ bị loại khỏi kết quả.",
        disabled=bool(_load_error),
    )

if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

analyze_clicked = st.button(
    "Phân tích và Gợi ý nghề nghiệp",
    type="primary",
    use_container_width=True,
    disabled=bool(_load_error) or st.session_state.analyzing,
    key="analyze_btn",
)

if analyze_clicked:
    st.session_state.analyzing = True
    st.session_state.analysis_error = None
    st.session_state.analysis_result = None
    st.rerun()

if st.session_state.analyzing:
    st.markdown(
        '<div class="cg-run-status">⏳ Đang chạy phân tích… vui lòng đợi.</div>',
        unsafe_allow_html=True,
    )

    cleaned_bio = (user_bio or "").strip()

    if len(cleaned_bio) < MIN_TEXT_LENGTH:
        st.session_state.analyzing = False
        st.session_state.analysis_error = (
            f"Văn bản quá ngắn (tối thiểu {MIN_TEXT_LENGTH} ký tự). "
            "Hãy mô tả rõ hơn về sở thích, kỹ năng và kinh nghiệm của bạn."
        )
        st.rerun()

    assert model is not None and field_encoder is not None
    assert career_encoder is not None and skills_encoder is not None

    try:
        with st.spinner("Đang phân tích hồ sơ..."):
            raw_features = call_gemini_extractor(
                cleaned_bio,
                list_of_all_careers,
                list_of_all_fields,
                list_of_all_skills,
            )

        features = validate_features(
            raw_features,
            list_of_all_careers,
            valid_skills=list_of_all_skills,
        )

        # --- Gộp lựa chọn thủ công của user với kết quả Gemini ---
        if selected_field and selected_field != FIELD_AUTO:
            features["Field"] = selected_field

        gemini_skills = list(features.get("Skills") or [])
        features["Skills"] = list(dict.fromkeys([*selected_skills, *gemini_skills]))

        gemini_pref = list(features.get("Preferred_Careers") or [])
        gemini_excl = list(features.get("Excluded_Careers") or [])
        features["Preferred_Careers"] = list(
            dict.fromkeys([*selected_preferred, *gemini_pref])
        )
        features["Excluded_Careers"] = list(
            dict.fromkeys([*selected_excluded, *gemini_excl])
        )

        excl_set = {c.lower() for c in features["Excluded_Careers"]}
        features["Preferred_Careers"] = [
            c for c in features["Preferred_Careers"] if c.lower() not in excl_set
        ]

        if int(user_projects) > 0:
            features["Projects"] = int(user_projects)
        if int(user_internships) > 0:
            features["Internships"] = int(user_internships)

        preferred_careers = features.get("Preferred_Careers", [])
        excluded_careers = features.get("Excluded_Careers", [])

        vector, x_local, skill_list = build_feature_vector(
            features,
            field_encoder,
            skills_encoder,
            warning_callback=st.warning,
            scaler=(extras or {}).get("scaler") if extras else None,
        )
        top_careers = predict_top_careers(
            vector,
            model,
            career_encoder,
            excluded_careers=excluded_careers,
            preferred_careers=preferred_careers,
            top_k=5,
            x_local=x_local,
            field_name=str(features.get("Field", "")),
            user_skills=skill_list,
            extras=extras,
        )

        with st.spinner("AI đang phân tích độ phù hợp của bạn với từng ngành nghề..."):
            career_explanations = explain_top_careers_with_gemini(
                features, top_careers
            )

        st.session_state.analysis_result = {
            "features": features,
            "top_careers": top_careers,
            "career_explanations": career_explanations,
            "preferred_careers": preferred_careers,
            "excluded_careers": excluded_careers,
        }
        st.session_state.analysis_error = None
    except json.JSONDecodeError:
        st.session_state.analysis_error = (
            "Không parse được JSON từ Gemini "
            "(có thể bị lỗi format / markdown). "
            "Vui lòng thử lại hoặc diễn đạt rõ hơn."
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state.analysis_error = f"JSON trích xuất không hợp lệ: {exc}"
    except Exception as exc:  # noqa: BLE001
        st.session_state.analysis_error = f"Lỗi khi phân tích: {exc}"
    finally:
        st.session_state.analyzing = False
        st.rerun()

if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

if st.session_state.analysis_result and not st.session_state.analyzing:
    st.markdown(
        '<div class="cg-run-status done">✅ Phân tích hoàn tất.</div>',
        unsafe_allow_html=True,
    )
    result = st.session_state.analysis_result
    render_profile_dashboard(result["features"])
    st.markdown("---")
    render_top_careers(
        result["top_careers"],
        result["career_explanations"],
        preferred_careers=result["preferred_careers"],
        excluded_careers=result["excluded_careers"],
    )
