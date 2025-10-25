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
    <p>For a walkthrough of the project, watch the demo video below: (<a href="https://www.youtube.com/watch?v=JCv2n1I46uA" target="_blank">Youtube Link</a>)</p>
    <br/>
    <div style="text-align: center; margin: 1rem 0;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/JCv2n1I46uA" 
            title="SingN'Seek Demo Video" frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
            allowfullscreen style="max-width: 100%;"></iframe>
    </div>
    <br/>
    <br/>
    <h5 style="margin-top:0;">Demo currently unavailable (<a href="https://github.com/mmohanram13/SingNSeek" target="_blank">Go to Github Repo</a>)</h5>
    <p>The hosted demo is temporarily unavailable because the <b>Elasticsearch trial has ended</b>.</p>
    <p>Please follow the setup instructions in the project's <b>GitHub README to run the application locally</b>:</p>
    <p><a href="https://github.com/mmohanram13/SingNSeek" target="_blank">https://github.com/mmohanram13/SingNSeek</a></p>
    <p style="margin-bottom:0;">Thank you for your interest.</p>
</div>
"""

st.markdown(notice_html, unsafe_allow_html=True)

st.stop()
