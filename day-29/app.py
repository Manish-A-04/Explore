from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pyttsx3
import speech_recognition as sr
import os
from dotenv import load_dotenv
import whisper
import soundfile as sf
import io
import numpy as np
import time

model = whisper.load_model("base")

load_dotenv()

defaults = {
    "api_key":os.getenv("GOOGLE_API_KEY"),
    "model":"gemini-2.0-flash",
    "temperature":0.6,
    "voice":"com.apple.eloquence.en-US.Grandpa",
    'volume':1.0,
    "rate":200,
    "session_id":"00123",
    "ability":"Psychology",
}

llm = ChatGoogleGenerativeAI( model = defaults["model"] ,
                             temperature = defaults["temperature"],
                              api_key = defaults["api_key"] )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You're an assistant who's good at {ability}. Respond in 20 words or fewer"),
        ("ai","Hello I am Lyra. How can I help you today?"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}")
    ]
)

base_chain = prompt|llm
store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

tts_engine = pyttsx3.init()
tts_engine.setProperty("voice",defaults["voice"])
tts_engine.setProperty("volume",defaults["volume"])
tts_engine.setProperty("rate",defaults["rate"])

def speak(text):
    print("Lyra: ",text)
    tts_engine.say(text)
    tts_engine.runAndWait()
r = sr.Recognizer()

def listen():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5 
    recognizer.non_speaking_duration = 0.5 
    
    print("Listening...............")
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            recognizer.dynamic_energy_threshold = True
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
        print("Processing...............")
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        float_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        model = whisper.load_model("base")
        result = model.transcribe(float_audio)
        transcribed_text = result["text"].strip()
        
        print(f"Transcribed: {transcribed_text}")
        return transcribed_text
        
    except sr.WaitTimeoutError:
        print("No speech detected within timeout period")
        return None
    except Exception as e:
        print(f"Error during recording or transcription: {e}")
        return None
    
def generate_response(ability , prompt):
    completions = with_message_history.invoke(
        {"ability":ability,"input":prompt},
        config={"configurable":{"session_id":defaults["session_id"]}}
    )
    message = completions.content
    return message
speak("Hello , Iam Lyra. How can I help you today?")
flag = True
while True:
    option = input("Enter 1-start recording, 2-exit\n")
    if int(option)==1:
        print("Listening...........")
        flag = False
        prompt = listen()
        if prompt is not None:
            print("You: ",prompt)
            response = generate_response(ability=defaults["ability"],prompt = prompt)
            flag=True
            sentences = response.split(".")
            for sentence in sentences:
                speak(sentence)

        else:
            flag = True
            speak("I'm sorry , I didnt understand that")


