import os

from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

pdf_data_path = "data"
vector_db_path = "azure_openai/vector_stores/db_faiss"


def create_db_from_pdf():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"]
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db


if __name__ == "__main__":
    db = create_db_from_pdf()
