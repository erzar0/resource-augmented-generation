from langchain.vectorstores import Chroma
from src.app import SentenceTransformersEmbeddings, EMBEDDING_MODEL

embedding = SentenceTransformersEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory="chromadb", embedding_function=embedding)

all_documents = vectorstore._collection.get()

print(all_documents)
# Display retrieved documents
# for i, doc in enumerate(all_documents['documents']):
#     print(i)
#     print(doc)
#     print("\n")
