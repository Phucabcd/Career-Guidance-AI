"""
app.py
Giao diện Streamlit cho Career Guidance AI.
Chạy: streamlit run app.py
"""

import streamlit as st
from model import prediction_career
from data.suggestions import INTEREST_SUGGESTIONS

st.set_page_config(page_title="Career Guidance AI", page_icon="🧭")

st.caption("🧭 Career Guidance AI")
st.title("Tìm ngành học phù hợp với bạn")
st.write("Mô tả về sở thích, điểm mạnh và điều bạn quan tâm.")


selected = st.pills(
    "Gợi ý",
    INTEREST_SUGGESTIONS,
    selection_mode="multi"
)

describe = st.text_area(
    "Mô tả bản thân",
    value=", ".join(selected),
    placeholder="Em thích lập trình, giải toán, tìm hiểu cách máy tính hoạt động...",
    height=120,
    label_visibility="collapsed",
)

if st.button("✨ Gợi ý ngành nghề", type="primary", use_container_width=True):
    if not describe.strip():
        st.warning("Bạn hãy nhập mô tả về bản thân trước nhé.")
    else:
        with st.spinner("Đang phân tích hồ sơ..."):
            result = prediction_career(describe)

        st.subheader("Kết quả gợi ý")

        for r in result:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{r['career_name']}**")
                with col2:
                    st.markdown(f"`{r['match']}% Match`")

                st.markdown("**Kỹ năng / định hướng nổi bật**")

                skills = [s.strip() for s in r["des"].split(",") if s.strip()]
                
                for skill in skills[:6]:
                    st.markdown(f"✅ {skill}")
                    
                if len(skills) > 6:
                    with st.expander("Xem thêm"):
                        for skill in skills[6:]:
                             st.markdown(f"• {skill}")
                            
                # with st.expander("Vì sao gợi ý ngành này?"):
                #     st.write(f"Mô tả của bạn gần nhất với hồ sơ: **{r['career_name']}**, phát triển theo hướng: {r['des']}")
                    
                    
                # if r["tu_khoa_khop"]:
                    #     st.caption(
                    #         "Khớp nhờ các từ khóa: " + ", ".join(r["tu_khoa_khop"])
                    #     )
                    # else:
                    #     st.caption("Khớp dựa trên ý nghĩa tổng thể của mô tả (không có từ khóa trùng trực tiếp).")