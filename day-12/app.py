import PIL.Image
import pytesseract as ts
import PIL

ts.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = PIL.Image.open("bill.jpg")
res = ts.image_to_string(img)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200 , chunk_overlap = 20)
text_splits = text_splitter.split_text(res)



embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_db = Chroma.from_texts(text_splits , embedding_model)
retriever = vector_db.as_retriever()

llm = GoogleGenerativeAI(model="gemini-2.0-flash")
prompt = ChatPromptTemplate.from_template(
 """
You are an expert at all domains, answer the user queries
based on the given context. Make sure to extract meaningful insights from it.
Context:
{context}
Query:
{input}
"""
)
combine = create_stuff_documents_chain(llm=llm , prompt=prompt)
retriever_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine)

res = retriever_chain.invoke({"input":"What was the amount of rs paid"})
print(res)