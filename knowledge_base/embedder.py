from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def create_embeddimgs():
    pdfloader = PyPDFLoader('data\clinical_guidlines-33-574.pdf')
    docs = pdfloader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5",encode_kwargs={"normalize_embeddings": True})
    db = Chroma.from_documents(texts, embedding = embeddings, persist_directory="data/embeddings")

    retriever = db.as_retriever(search_kwargs={"k": 10})

    return retriever
