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
OLLAMA_API_URL = "http://ollama:11434/api"
OLLAMA_MODEL = "library/qwen2.5:3b"

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

def query_vector_db(query: str, selected_collection, top_k: int = 5) -> List[str]:
    """Query the locally persisted Chroma vector database for relevant documents."""
    try:
        vectorstore = Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding, collection_name=selected_collection)
        results: List[Document] = vectorstore.similarity_search(query, k=top_k)
        return [result.page_content for result in results]
    except Exception as e:
        st.error(f"Error querying the vector database: {e}")
        return []

def upload_and_vectorize_file(file, selected_collection) -> str:
    """Upload a .txt file and vectorize its content with a loading bar and document count."""
    vectorstore = Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding, collection_name=selected_collection)
    try:
        content = file.read().decode("utf-8")

        texts = list(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(content))

        st.write("Vectorizing text...")
        progress_bar = st.progress(0)
        total_texts = len(texts)

        doc_count_placeholder = st.empty()

        for i, doc in enumerate(texts):
            vectorstore.add_texts([doc])

            progress_bar.progress(int((i + 1) / total_texts * 100))
            doc_count_placeholder.write(f"Processed {i + 1}/{total_texts} documents")

        progress_bar.progress(100)
        doc_count_placeholder.write("Vectorization complete.")

        return "File successfully vectorized!"
    except Exception as e:
        vectorstore.delete_collection()
        return f"Error processing file: {e}"

def generate_response_with_ollama(prompt: str) -> Generator[any, any, any]:
    """Generate a response using Ollama's API."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
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


st.set_page_config(page_title="LekturR", layout="wide", page_icon="🔍")

_, col1, _ = st.columns([1, 4, 1])

with col1:
    st.title("🔍 LekturR")
    st.markdown(
        "Welcome to **LekturR**! Upload your file, vectorize it, and search for relevant information."
    )

    st.sidebar.header("📜 Select LLM")
    selected_model = st.sidebar.radio(
        "Bigger models require more memory. Make sure to select a model suitable for your computer!",
        ["Qwen2.5", "Bielik-11B-v2.3-Instruct"],
        captions=[
            "1.9 GB",
            "6.7 GB",
        ],
    )
    if selected_model == "Qwen2.5":
        OLLAMA_MODEL = "library/qwen2.5:3b"
    else:
        OLLAMA_MODEL = "SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M"

    st.sidebar.header("🗒️ Select Collection")
    st.sidebar.markdown("Select collection or create a new one to store similar documents")
    vectorstore = Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding)
    collections = vectorstore._client.list_collections()
    collection_names = [x.name for x in collections if x.name != 'langchain']

    selected_collection = st.sidebar.selectbox("Select collection", collection_names)

    st.sidebar.markdown('or')

    collection_name = st.sidebar.text_input("Create new collection", key="create_collection")
    if st.sidebar.button("Create Collection"):
        if collection_name:
            collection_name = collection_name.lower().replace(" ", "_")
            Chroma(persist_directory=VECTORSTORE_NAME, embedding_function=embedding, collection_name=collection_name)
            st.sidebar.success(f"Collection {collection_name} created.")
        else:
            st.sidebar.error("Please enter a collection name before creating it.")

    st.sidebar.header("📂 Upload Your File")
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file to vectorize:", type=["txt"])

    if st.sidebar.button("Vectorize File"):
        if uploaded_file:
            result = upload_and_vectorize_file(uploaded_file, selected_collection)
            st.sidebar.success(result)
        else:
            st.sidebar.error("Please upload a .txt file before clicking the button.")


    st.header("💡 Query Relevant Information")
    query = st.text_input("Enter your prompt:")
    top_k = st.slider("🔍 Number of documents to retrieve:", 1, 5, 1)

    if st.button("Submit Query"):
        if query:
            st.markdown("### 🔄 Querying relevant documents...")

            relevant_docs = query_vector_db(query, selected_collection, top_k)

            if relevant_docs:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📄 Retrieved Documents")
                    for i, doc in enumerate(relevant_docs, start=1):
                        color = "#222222" if i % 2 == 0 else "#444444"
                        styled_doc = f"""
                        <div style='background-color: {color}; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                            <strong>Document {i}:</strong> {doc}
                        </div>
                        """
                        st.markdown(styled_doc, unsafe_allow_html=True)

                with col2:
                    st.subheader("🤖 AI Response")

                    with st.spinner("Generating response... Please wait."):
                        context = "\n".join(relevant_docs)
                        prompt = f"""
                                Given the following context:\n{context}
                                Answer the following question:\n{query}
                                If the context contains enough information, answer the question directly. 
                                If it does not contain enough information or is irrelevant, say so and ask for more details. 
                                Do not provide answers to irrelevant questions.
                                Answer:
                                """
                        st.write_stream(generate_response_with_ollama(prompt))

            else:
                st.error("No relevant documents found for your query.")
                prompt = f"Notify the user that no related documents were found for their query: {query}."
                st.write_stream(generate_response_with_ollama(prompt))
        else:
            st.warning("Please enter a query to proceed.")
