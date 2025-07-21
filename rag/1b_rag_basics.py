import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Define Gemini embedding wrapper
class GeminiEmbeddings:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyA-KyS4dzHvlSwmQ8quIHsT_AYA5EvmbBI",  # ⛔ Replace this with your actual API key
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)


# Initialize Gemini embeddings
embeddings = GeminiEmbeddings()

# Define the path to the persistent Chroma DB
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load Chroma vector store with Gemini embeddings
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define user query
query = "Who is Odysseus' wife?"

# Set up retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)

# Retrieve relevant documents
relevant_docs = retriever.invoke(query)

# Print the results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
