import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def scrape_medium_article(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        await page.wait_for_selector('article')

        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        title = soup.find('h1').get_text()

        article = soup.find('article')
        paragraphs = article.find_all('p')
        article_text = '\n'.join([para.get_text() for para in paragraphs])

        ol_list = article.find_all("ol")
        ol_text = "\n".join([text.get_text() for text in ol_list ])

        ul_list = article.find_all("ul")
        ul_text = "\n".join([text.get_text() for text in ul_list ])


        return title, article_text , ol_text , ul_text

url = 'https://www.geeksforgeeks.org/large-language-model-llm/'
title, content , text  , ul = asyncio.run(scrape_medium_article(url))

combined = [title , content , text , ul]

for i in combined:
    with open("extract.txt" , "w") as f:
        f.write(i)



from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
loader = TextLoader("extract.txt")
combined = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
splits = splitter.split_documents(combined)

embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
db = Chroma.from_documents(splits , embeddings)
retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
You are an expert assistant .
Answer the users queries based on the given context.

Context:
{context}

Question:
{input}
"""
)

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

combine_chain = create_stuff_documents_chain(llm = llm , prompt=prompt)

retrieval_chain = create_retrieval_chain(retriever=retriever , combine_docs_chain=combine_chain)
inp = "what is the difference between nlp and llm"
res = retrieval_chain.invoke({"input" : inp})

print(res)


