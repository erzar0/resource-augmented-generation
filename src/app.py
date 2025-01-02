import os
import requests
import json
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from typing import List, Generator

VECTORSTORE_NAME = "chromadb"
EMBEDDING_MODEL = "OrlikB/KartonBERT-USE-base-v1"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

class SentenceTransformersEmbeddings(Embeddings):
    """Implementation of the Embeddings interface using Sentence Transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding model using Sentence Transformers."""
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using Sentence Transformers."""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using Sentence Transformers."""
        return self.model.encode([text], convert_to_tensor=False)[0]

embedding = SentenceTransformersEmbeddings(model_name=EMBEDDING_MODEL)

def query_vector_db(query: str, top_k: int = 5) -> List[str]:
    """Query the locally persisted Chroma vector database for relevant documents."""
    try:
        vectorstore = Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding)
        results = vectorstore.similarity_search(query, k=top_k)
        return [result["text"] for result in results]
    except Exception as e:
        st.error(f"Error querying the vector database: {e}")
        return []

def upload_and_vectorize_file(file) -> str:
    """Upload a .txt file and vectorize its content with a loading bar."""
    try:
        content = file.read().decode("utf-8")

        print(content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        documents = [Document(page_content=text) for text in text_splitter.split_text(content)]
        print(documents)
        
        vectorstore = Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding)

        st.write("Vectorizing text...")
        progress_bar = st.progress(0)
        total_docs = len(documents)
        for i, doc in enumerate(documents):
            vectorstore.add_documents([doc])
            progress_bar.progress(int((i + 1) / total_docs * 100))
        
        vectorstore.persist()
        progress_bar.progress(100)

        return f"File successfully vectorized and stored in {VECTORSTORE_NAME}."
    except Exception as e:
        return f"Error processing file: {e}"

def generate_response_with_ollama(prompt: str) -> Generator[any, any, any]:
    """Generate a response using Ollama's API."""
    try:
        response = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True
        )

        if response.status_code == 200:
            response_text = ""
            for chunk in response.iter_lines():
                json_response = json.loads(chunk)
                if json_response["done"] == False:
                    yield json_response["response"]
            return response_text
        else:
            st.error(f"Error generating response: {response.text}")
            return ""
    except Exception as e:
        st.error(f"Error communicating with Ollama API: {e}")
        return ""

st.title("🔍 LekturR")

uploaded_file = st.file_uploader("Upload a .txt file to vectorize:", type=["txt"])
if uploaded_file:
    result = upload_and_vectorize_file(uploaded_file)
    st.write(result)

query = st.text_input("Enter your prompt:")
top_k = st.slider("Number of documents to retrieve:", 1, 10, 3)

if query:
    st.write("### Querying relevant documents...")
    relevant_docs = query_vector_db(query, top_k)

    if relevant_docs:
        st.write("### Retrieved Documents:")
        for i, doc in enumerate(relevant_docs, start=1):
            st.write(f"**Document {i}:** {doc}")

        context = "\n".join(relevant_docs)
        prompt = f"Given the following context:\n{context}\n\nAnswer the following question:\n{query}"

        st.write_stream(generate_response_with_ollama(prompt))
    else:
        st.write("### Failed To Retrieve Documents.")
        prompt = f"Say to the user that no related documents were found for their query: {query}."
        
        st.write_stream(generate_response_with_ollama(prompt))
