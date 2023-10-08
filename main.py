import os 
os.environ["OPENAI_API_KEY"] = "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"
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

if __name__ == "__main__":
    print("Hi")

    pdf_path = r"C:\Users\ejose\Documents\langchain\local-vector-store\2210.03629.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")