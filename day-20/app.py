import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="centered")
st.title("Chatbot using Langchain")
st.markdown("### Gemini-2.0-flash")

if "messages" not in st.session_state:
    st.session_state["messages"] = [SystemMessage(content="You are a helpful assistant.")]

for msg in st.session_state["messages"]:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for chunk in llm.stream(st.session_state["messages"]):
            full_response += chunk.content
            response_container.markdown(full_response)
    
    st.session_state["messages"].append(AIMessage(content=full_response))