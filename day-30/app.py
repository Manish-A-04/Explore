import streamlit as st
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper 
from langchain_community.tools import WolframAlphaQueryRun
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
load_dotenv()


gemini = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )
wolfram = WolframAlphaAPIWrapper()
wolfram = WolframAlphaQueryRun(api_wrapper=wolfram )

st.title('Startup Idea Validator')
st.markdown('Analyze your startup idea, get SWOT analysis, business model suggestions, and customer personas.')

idea = st.text_area('Enter your startup idea:')
industry = st.text_input('Industry:')
target_audience = st.text_input('Target Audience:')

if st.button('Validate Idea'):
    feasibility_prompt = PromptTemplate(input_variables=['idea', 'industry'], template='Analyze the feasibility of {idea} in the {industry} industry.')
    feasibility_chain = LLMChain(llm=gemini, prompt=feasibility_prompt)
    feasibility = feasibility_chain.run(idea=idea, industry=industry)

    swot_prompt = PromptTemplate(input_variables=['idea'], template='Perform a SWOT analysis for the startup idea: {idea}.')
    swot_chain = LLMChain(llm=gemini, prompt=swot_prompt)
    swot_analysis = swot_chain.run(idea=idea)

    business_prompt = PromptTemplate(input_variables=['idea'], template='Suggest a business model for the startup idea: {idea}.')
    business_chain = LLMChain(llm=gemini, prompt=business_prompt)
    business_model = business_chain.run(idea=idea)

    market_data = wolfram.run(f'current market trends in {industry}')

    st.subheader('Feasibility Assessment')
    st.write(feasibility)
    st.subheader('SWOT Analysis')
    st.write(swot_analysis)
    st.subheader('Business Model Suggestions')
    st.write(business_model)
    st.subheader('Market Analysis')
    st.write(market_data)



