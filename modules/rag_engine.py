from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import os
import shutil


def create_vector_store(text):

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create temporary clean database path
    persist_directory = tempfile.mkdtemp()

    # Create Chroma vector store
    db = Chroma.from_texts(
        chunks,
        embeddings,
        persist_directory=persist_directory
    )

    return db