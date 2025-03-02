import os
import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

st.title("📚 RAG Chatbot with LangChain")
st.markdown("Ask questions about your documents using Gemini and ChromaDB")

with st.sidebar:
    st.header("Configuration")
        
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    
    process_doc_button = st.button("Process Documents")
    
    if st.session_state.documents_processed:
        st.success("Documents processed! You can now ask questions.")

def process_documents(files):
    documents = []
    
    for file in files:
        file_path = f"./temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            
        os.remove(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    
    return vectorstore

def initialize_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )
    
    return chain

if process_doc_button and uploaded_files:
    with st.spinner("Processing documents..."):
        vectorstore = process_documents(uploaded_files)
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_processed = True
        
        st.session_state.chain = initialize_rag_chain(vectorstore)

st.subheader("Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Ask a question about your documents...")

if user_query :
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
    
    if not st.session_state.documents_processed:
        with st.chat_message("assistant"):
            st.write("Please upload and process documents first using the sidebar options.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process documents first using the sidebar options."})
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"question": user_query})
            answer = response["answer"]
        
        with st.chat_message("assistant"):
            st.write(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()