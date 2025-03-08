from fastapi import FastAPI
from langserve import add_routes
from langchain.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="LangServe RAG API", version="1.0")

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

loader = TextLoader("sample.txt")  
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embedding_model  = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
vector_store = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

add_routes(app, qa_chain, path="/rag")

