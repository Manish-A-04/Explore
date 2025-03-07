from langchain.agents import AgentType, initialize_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import dotenv
dotenv.load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def collect_symptoms(user_input):
    return f"Noted symptoms: {user_input}. Are there any additional symptoms?"

symptom_tool = Tool(
    name="Symptom Collection Tool",
    func=collect_symptoms,
    description="Collects user symptoms for diagnosis."
)

def diagnose(symptoms):
    prompt = f"Given the symptoms: {symptoms}, what are the possible medical conditions?"
    return llm.predict(prompt)

diagnosis_tool = Tool(
    name="Diagnosis Tool",
    func=diagnose,
    description="Suggests possible medical conditions based on symptoms."
)

def assess_severity(symptoms):
    prompt = f"Based on the symptoms: {symptoms}, classify the severity as Mild, Moderate, or Severe."
    return llm.predict(prompt)

severity_tool = Tool(
    name="Severity Assessment Tool",
    func=assess_severity,
    description="Assesses the severity of the symptoms."
)

def suggest_medication(symptoms):
    prompt = f"For the symptoms: {symptoms}, suggest over-the-counter medication and home remedies."
    return llm.predict(prompt)

medication_tool = Tool(
    name="Medication Advice Tool",
    func=suggest_medication,
    description="Suggests general medication and home care based on symptoms."
)

agent = initialize_agent(
    tools=[symptom_tool, diagnosis_tool, severity_tool, medication_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

print("Welcome to the Medical Assistant Chatbot. Describe your symptoms.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! Take care.")
        break
    response = agent.run(user_input)
    print("Bot:", response)