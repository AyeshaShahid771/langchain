import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# === Gemini Embedding Wrapper ===
class GeminiEmbeddings:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyA-KyS4dzHvlSwmQ8quIHsT_AYA5EvmbBI",  # <-- Replace this with your real Gemini API key
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)


# === Define directories ===
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")  # Directory containing .txt files
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# === Check if vector store exists ===
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Check if the books folder exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # Load all .txt files as documents
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")  # Avoid encoding errors
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # === Split into smaller chunks ===
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # === Create Gemini embeddings ===
    print("\n--- Creating Gemini embeddings ---")
    embeddings = GeminiEmbeddings()
    print("--- Finished creating embeddings ---")

    # === Save to Chroma vector store ===
    print("\n--- Creating and persisting Chroma vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
