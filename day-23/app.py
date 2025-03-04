import os
from dotenv import load_dotenv
import supabase
from langchain.agents import create_sql_agent
from langchain.utilities import SQLDatabase
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import urllib
load_dotenv()


PASSWORD = os.getenv("PASSWORD")
password =  urllib.parse.quote(PASSWORD)
database = SQLDatabase.from_uri(f"postgresql://postgres:{password}@db.fuuvplgujfmkhuuehcoo.supabase.co:5432/postgres")

llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )

agent = create_sql_agent(
    llm = llm,
    db = database , 
    verbose = True ,
    handle_parsing_errors = True
)


if __name__ == "__main__":
    while True:
        inp = input("Ask me about room's available , booking and available menus: \n")
        if inp.lower()=="exit":
            break
        response = agent.invoke({"input":inp})
        print(response["output"])
