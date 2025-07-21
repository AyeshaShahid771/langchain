import os

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# === Custom Gemini Wrapper ===
class GeminiEmbeddings:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyA-KyS4dzHvlSwmQ8quIHsT_AYA5EvmbBI",  # Replace this with your actual API key
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)


# === Set paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# === Initialize Gemini embeddings ===
embeddings = GeminiEmbeddings()

# === Load existing Chroma vector store ===
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# === User query ===
query = "How did Juliet die?"

# === Retrieve top 3 relevant docs with score threshold ===
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.7},  # You can tune this threshold
)
relevant_docs = retriever.invoke(query)

# === Display results ===
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
