import os
import hashlib
import pandas as pd
from docx import Document
import PyPDF2
import cohere
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

def extract_text_from_file(path):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
        return ["\n".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    
    elif path.endswith(".txt"):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().split('\n\n')
    
    elif path.endswith(".docx"):
        doc = Document(path)
        return [para.text for para in doc.paragraphs if para.text.strip()]
    
    elif path.endswith(".pdf"):
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() for page in reader.pages if page.extract_text()]
    
    else:
        raise ValueError("Unsupported file format")

def get_text_hash(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)

    def embed_documents(self, texts):
        return self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings

    def embed_query(self, text):
        return self.client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

def main():
    API_KEY = "x6OWAtFharUcnAk3IJoGUgKcks2k5N2h6pojfNqR"
    FILE_PATH = r"D:\rag\data\Top 10 Chocolate Bars.pdf"
    PERSIST_DIR = "./chroma_excel_semantic_db"

    print(f"Loading file: {FILE_PATH}")
    chunks = extract_text_from_file(FILE_PATH)
    print(f"Extracted {len(chunks)} text chunks.\n")

    embedding_model = CohereEmbeddings(API_KEY)

    if os.path.exists(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    else:
        vectorstore = Chroma.from_texts(texts=[], embedding=embedding_model, persist_directory=PERSIST_DIR)

    existing_hashes = set(get_text_hash(doc.page_content) for doc in vectorstore.similarity_search("", k=1000))
    
    new_chunks = []
    for chunk in chunks:
        if get_text_hash(chunk) not in existing_hashes:
            new_chunks.append(chunk)
        else:
            print("Duplicate chunk skipped.")

    if new_chunks:
        print(f"Adding {len(new_chunks)} new unique chunks to vector store...\n")
        vectorstore.add_texts(new_chunks)
        vectorstore.persist()
    else:
        print("No new unique chunks to add.")

    query = input("Enter your question: ").strip()
    
    try:
        k = int(input("How many top results do you want to retrieve? (Default: 5): ") or "5")
    except ValueError:
        k = 5
        print("Invalid input. Defaulting to 5.")

    retrieved_docs = vectorstore.similarity_search(query, k=k)

    print(f"\nTop-{k} Retrieved Chunks:\n")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content.strip()}\n")

    co = cohere.Client(API_KEY)
    response = co.chat(
        model="command-a-03-2025",
        message=query,
        documents=[{"text": doc.page_content} for doc in retrieved_docs]
    )

    print("\nCohere Response:\n", response.text)

if __name__ == "__main__":
    main()