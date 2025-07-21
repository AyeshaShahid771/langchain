import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# === Load environment variables ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# === Define custom Gemini embeddings class ===
class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/embedding-001"):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model=self.model_name, content=text, task_type="retrieval_document"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_query"
        )["embedding"]


# === Set up paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# === If Chroma DB doesn't exist, build vector store ===
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load file with UTF-8 encoding
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create Gemini embeddings
    print("--- Creating Gemini embeddings ---")
    embeddings = GeminiEmbeddings()
    print("--- Finished creating embeddings ---")

    # Create vector store and persist it
    print("--- Creating Chroma vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
