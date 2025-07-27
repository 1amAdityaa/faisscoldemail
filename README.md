# 🚀 Cold Email Generator

Turn job descriptions into personalized cold emails — powered by **LLaMA-3** (via Together AI), **Streamlit**, **LangChain**, and **ChromaDB**.

![App Screenshot](https://github.com/1amAdityaa/updated-cold-email/blob/main/coldemail.png)

---

## 🔥 What It Does

Paste any job posting URL and this app will:
1. 🔍 Scrape & clean the job description
2. 🤖 Extract job info (role, experience, skills, description) using LLaMA-3
3. 🎯 Match relevant portfolio links based on required skills
4. 💌 Generate a personalized cold email ready to send

---

## 🧠 Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – for the frontend
- [LangChain](https://www.langchain.com/) – for chaining LLM prompts
- [Together AI](https://www.together.ai/) – runs LLaMA-3 model
- [ChromaDB](https://www.trychroma.com/) – for vector-based skill-link matching
- [Pandas](https://pandas.pydata.org/) – for handling CSV portfolio

---

## 📁 Project Structure

```
coldemail/
├── app.py               # 🚀 Streamlit UI
├── chains.py            # 🤖 LLM prompt logic
├── portfolio.py         # 🔗 Skill → link search
├── utils.py             # 🧼 clean_text function
├── .env                 # 🔐 API key for Together
├── requirements.txt     # 📦 Python dependencies
└── app/
    └── resources/
        └── my_data.csv  # 📁 Your portfolio links
```

---

## ⚙️ Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/yourusername/coldemailgenerator.git
cd coldemailgenerator
```

### 2. Create `.env` file

```env
TOGETHER_API_KEY=your-real-key-here
```

> Get your key from [https://together.ai/auth/api-keys](https://together.ai/auth/api-keys)

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your portfolio data

Create this file:

```bash
app/resources/my_data.csv
```

And fill it like:

```csv
Techstack,link
Python,https://1-am-aditya.netlify.app/
React,https://1-am-aditya.netlify.app/
Java,https://1-am-aditya.netlify.app/
```

---

## ▶️ Run the App Locally

```bash
streamlit run app.py
```

App will open at: `http://localhost:8501`

---

## 🌐 Deploying to Streamlit Cloud

1. Push this project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Paste repo link
4. Set `TOGETHER_API_KEY` in **Secrets Manager**
5. Click Deploy 💥

---

## ✨ Features

- 🔗 Paste job URLs (career pages, etc.)
- 🧠 Extracts job info using LLaMA-3
- 🧰 Smart matching from your portfolio
- 💌 Cold email output in Markdown
- 📱 Responsive UI with collapsible sections

---

## 📸 Screenshot

![App Screenshot](https://via.placeholder.com/800x400.png?text=Cold+Email+Generator+UI)

---

## 📄 License

MIT License  
© 2025 [Aditya Raj](https://1-am-aditya.netlify.app)
