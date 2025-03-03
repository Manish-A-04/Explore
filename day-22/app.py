from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableBranch , RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )

ai_check_prompt = ChatPromptTemplate.from_template(
    """
    Determine if the following comment is written by an AI or a human:
    "{comment}"
    Respond with only "AI" or "Human".
"""
)

ai_prompt ="""
    The given comment is AI generated so return a warning feedback indicating that bot comments are not allowed.
    Mention strictly that bot comments are not allowed anf if found repeatedly account may be banned.
"""
ai_response_prompt = RunnableLambda( lambda x : ai_prompt)

sentiment_check_prompt = ChatPromptTemplate.from_template(
    """
    Analyze the sentiment of the following comment carefully , try to uderstand its intent properly and classify it as "Positive", "Neutral", or "Negative":
    "{comment}"
"""
)

response_postitve_prompt = ChatPromptTemplate.from_template(
    """
    Generate an response considering that the comment sentiment is positive.
    The company name is HOWDWAY generate a professional response in 30 - 80 words on behalf of it.
    Give just the feedback response nothing else
    Comment:
    {comment}

"""
)
response_neutral_prompt = ChatPromptTemplate.from_template(
    """
    Generate an response considering that the comment sentiment is neutral.
    The company name is HOWDWAY generate a professional response in 30 - 80 words on behalf of it.
    Give just the feedback response nothing else
    Comment:
    {comment}
"""
)
response_negative_prompt = ChatPromptTemplate.from_template(
    """
    Generate an response considering that the comment sentiment is negative.
    The company name is HOWDWAY generate a professional response in 30 - 80 on behalf of it.
    Give just the feedback response nothing else
    Comment:
    {comment}
    """
)
ai_check_chain = ai_check_prompt | llm | StrOutputParser()

sentiment_check_chain = sentiment_check_prompt | llm |StrOutputParser()
ai_response_chain = ai_response_prompt | llm | StrOutputParser()



response_positive_chain = response_postitve_prompt | llm | StrOutputParser()
response_neutral_chain = response_neutral_prompt | llm | StrOutputParser()
response_negative_chain = response_negative_prompt | llm | StrOutputParser()

sentiment_branch = RunnableBranch(
    ( lambda x : "positive" in x.lower()  , response_positive_chain),
    ( lambda x : "neutral" in x.lower() , response_neutral_chain),
    response_negative_chain
)
sentiment_chain = sentiment_check_chain | sentiment_branch

ai_check_branch = RunnableLambda( lambda x : sentiment_chain if "human" in x.lower() else ai_response_chain )

chain = ai_check_chain | ai_check_branch  

inp = input("Enter the comment : \n")
print(chain.invoke({"comment":inp}))