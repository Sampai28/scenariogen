import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"Key loaded: {'YES' if api_key else 'NO'}")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
response = llm.invoke("What is an AV simulation scenario? One sentence.")
print(response.content)