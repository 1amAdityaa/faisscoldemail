
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

class Portfolio:
    def __init__(self, file_path="app/resources/my_data.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.db = None

    def load_portfolio(self):
        documents = []
        for _, row in self.data.iterrows():
            techs = row["Techstack"].split()
            for tech in techs:
                documents.append(Document(page_content=tech, metadata={"links": row["link"]}))
        self.db = FAISS.from_documents(documents, self.embeddings)

    def query_links(self, skills):
        if not self.db:
            raise ValueError("Database not loaded. Call load_portfolio() first.")
        results = self.db.similarity_search_with_score(' '.join(skills), k=1)
        return [res[0].metadata for res in results if res]
