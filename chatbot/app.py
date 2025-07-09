from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set environment variables with proper error handling
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Langsmith Tracing
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's query in a concise and helpful manner."),
    ("user", "Question: {question}"),
])

# Streamlit Framework

st.title("LangChain Chatbot with Google Generative AI")

# Check if Google API key is available
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ Google API key not found! Please set your GOOGLE_API_KEY in the .env file or as an environment variable.")
    st.stop()

input_text = st.text_input("Search the topic you want to know about")

# Google Generative AI Model
LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
OUTPUT_PARSER = StrOutputParser()
chain = prompt_template | LLM | OUTPUT_PARSER

if input_text:
    st.write(chain.invoke({"question": input_text}))
