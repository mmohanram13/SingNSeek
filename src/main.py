import streamlit as st
from pathlib import Path

# Minimal demo page: logo + title, notice, and stop
st.set_page_config(page_title="SingN'Seek", page_icon="images/logo.png", layout="centered")

logo_path = Path(__file__).parent.parent / "images" / "logo.png"
if logo_path.exists():
    col_logo, col_title = st.columns([0.6, 3])
    with col_logo:
        st.image(str(logo_path), width=64)
    with col_title:
        st.markdown("<h1 style='margin:0;'>SingN'Seek</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>SingN'Seek</h1>", unsafe_allow_html=True)

st.markdown("---")

notice_html = """
<div style="max-width:900px;margin:2rem auto;padding:1.25rem;border-radius:8px;
            border:1px solid #e6e6e6;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
  <h3 style="margin-top:0;">Demo currently unavailable</h3>
  <p>The hosted demo is temporarily unavailable because the <b>Elasticsearch trial has ended.<b></p>
  <p>Please follow the setup instructions in the project's GitHub README to run the application locally:</p>
  <p>👉 <a href="https://github.com/mmohanram13/SingNSeek" target="_blank">https://github.com/mmohanram13/SingNSeek</a></p>
  <p>For a walkthrough of the project, watch the demo video on YouTube:</p>
  <p>👉 <a href="https://www.youtube.com/watch?v=JCv2n1I46uA" target="_blank">Watch the demo on YouTube</a></p>
  <p style="margin-bottom:0;">Thank you for your interest.</p>
</div>
"""

st.markdown(notice_html, unsafe_allow_html=True)

# Ensure no other app code runs on the hosted demo
st.stop()
import streamlit as st
from pathlib import Path

# Minimal demo page: logo + title, notice, and stop
st.set_page_config(page_title="SingN'Seek", page_icon="images/logo.png", layout="centered")

logo_path = Path(__file__).parent.parent / "images" / "logo.png"
if logo_path.exists():
    col_logo, col_title = st.columns([0.6, 3])
    with col_logo:
        st.image(str(logo_path), width=64)
    with col_title:
        st.markdown("<h1 style='margin:0;'>SingN'Seek</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>SingN'Seek</h1>", unsafe_allow_html=True)

st.markdown("---")

notice_html = """
<div style="max-width:900px;margin:2rem auto;padding:1.25rem;border-radius:8px;
            border:1px solid #e6e6e6;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
  <h3 style="margin-top:0;">Demo currently unavailable</h3>
  <p>The hosted demo is temporarily unavailable because the Elasticsearch trial has ended.</p>
  <p>Please follow the setup instructions in the project's GitHub README to run the application locally:</p>
  <p><a href="https://github.com/mmohanram13/SingNSeek" target="_blank">https://github.com/mmohanram13/SingNSeek</a></p>
  <p>For a walkthrough of the project, watch the demo video on YouTube:</p>
  <p><a href="https://www.youtube.com/watch?v=JCv2n1I46uA" target="_blank">Watch the demo on YouTube</a></p>
  <p style="margin-bottom:0;">Thank you for your interest.</p>
</div>
"""

st.markdown(notice_html, unsafe_allow_html=True)

st.stop()
