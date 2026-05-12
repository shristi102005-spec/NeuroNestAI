from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

def create_vector_store(text):

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector database
    db = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return db