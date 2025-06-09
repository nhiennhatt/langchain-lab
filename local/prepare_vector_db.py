from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings

pdf_data_path = "data"
vector_db_path = "local/vector_stores/db_faiss"

def create_db_from_pdf():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = LlamaCppEmbeddings(model_path="models/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2.Q8_0.gguf")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db

if __name__ == "__main__":
    db = create_db_from_pdf()
