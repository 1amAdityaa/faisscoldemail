import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains_fixed import Chain
from portfolio import Portfolio
from utils import clean_text

st.set_page_config(layout="wide", page_title="📧 Cold Email Generator", page_icon="💌")

# ------------------ 💅 Custom CSS Styling ------------------
st.markdown("""
    <style>
        body {
            background: #0f2027;
        }
        .main {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 3.5em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            color: white;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            margin-bottom: 30px;
            color: #ddd;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            padding: 0.75em;
            font-size: 1.1em;
            background-color: #1e2a38;
            color: white;
            border: 1px solid #444;
        }
        .stButton > button {
            background: linear-gradient(to right, #56ccf2, #2f80ed);
            color: white;
            padding: 0.7em 2em;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1em;
            transition: 0.2s ease-in-out;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #2f80ed, #56ccf2);
        }
        .highlight-box {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1.5em;
            border-radius: 1em;
            margin-bottom: 2em;
        }
        code {
            font-size: 0.95em;
            background-color: rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ 🚀 Title + Subtitle ------------------
st.markdown('<div class="title">🚀 Cold Email Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turn job posts into job offers — powered by LLaMA-3 🔥</div>', unsafe_allow_html=True)

# ------------------ 🔗 Input ------------------
with st.container():
    url_input = st.text_input("🔗 Paste Job URL Below:", placeholder="e.g. https://careers.xyz.com/job/software-intern")
    generate = st.button("✨ Generate Cold Email")

# ------------------ 🧠 Process Logic ------------------
if generate:
    llm = Chain()
    portfolio = Portfolio()
    with st.spinner("🛠 Scraping & analyzing job description..."):
        try:
            loader = WebBaseLoader([url_input])
            page_content = loader.load().pop().page_content
            data = clean_text(page_content)

            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            st.success(f"✅ Found {len(jobs)} job(s)")

            for idx, job in enumerate(jobs, 1):
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)

                with st.expander(f"📌 Job #{idx}: {job.get('role', 'Unknown Role')}"):
                    st.markdown(f"**Experience**: {job.get('experience', 'N/A')}  \n**Skills**: {', '.join(skills)}")
                    st.markdown(f"**Description**: {job.get('description', 'N/A')}")

                with st.expander("💌 Cold Email"):
                    st.code(email, language='markdown')

        except Exception as e:
            st.error(f"🚨 Something went wrong: {e}")
