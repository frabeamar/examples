from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv(".env")

def load_gemini_chat():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def load_ollama_chat():
    return ChatOllama(model = "llama3.1")
