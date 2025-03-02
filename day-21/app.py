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
# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# UI Header
st.title("📚 RAG Chatbot with LangChain")
st.markdown("Ask questions about your documents using Gemini and ChromaDB")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Configuration")
        
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    
    # Document processing button
    process_doc_button = st.button("Process Documents")
    
    # Show processing status
    if st.session_state.documents_processed:
        st.success("Documents processed! You can now ask questions.")

# Document processing function
def process_documents(files):
    documents = []
    
    for file in files:
        # Save uploaded file temporarily
        file_path = f"./temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Load based on file type
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            
        # Clean up temp file
        os.remove(file_path)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    
    return vectorstore

# Initialize Gemini model and RAG chain
def initialize_rag_chain(vectorstore):
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
    )
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )
    
    return chain

# Process documents when button is clicked
if process_doc_button and uploaded_files:
    with st.spinner("Processing documents..."):
        vectorstore = process_documents(uploaded_files)
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_processed = True
        
        # Initialize the RAG chain
        st.session_state.chain = initialize_rag_chain(vectorstore)

# Display chat interface
st.subheader("Chat with your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about your documents...")

# Handle user input
if user_query :
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Check if documents have been processed
    if not st.session_state.documents_processed:
        with st.chat_message("assistant"):
            st.write("Please upload and process documents first using the sidebar options.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process documents first using the sidebar options."})
    else:
        # Get response from chain
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"question": user_query})
            answer = response["answer"]
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(answer)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
elif user_query :
    # Remind user to input API key
    with st.chat_message("assistant"):
        st.write("Please enter your Google API Key in the sidebar first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please enter your Google API Key in the sidebar first."})

# Add a reset button in the sidebar
with st.sidebar:
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()