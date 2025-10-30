import streamlit as st
import os
import tempfile
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pytesseract
from PIL import Image
import json
from datetime import datetime

# -----------------------------
# 🔐 Groq API Key (Set here)
# -----------------------------
import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to chunk the text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a VectorStore from chunks
def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = Chroma.from_documents(documents, embedding)
    return vector_store

# Function to call Groq API
def chat_with_groq(prompt, model="llama3-70b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 1024,
        "stop": None,
        "stream": False
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# Summarize the PDF content
def summarize_text(text):
    return chat_with_groq(f"Summarize the following text:\n{text}")

# -----------------------------
# Streamlit Interface
# -----------------------------
st.set_page_config(page_title="PDF Chat with Groq", page_icon="📚")
st.title("📚 PDF Chat with Groq Llama3.3 🤖")

# Sidebar UI
with st.sidebar:
    st.header("📂 Upload & Settings")
    pdf_files = st.file_uploader("Upload PDFs 📄", type="pdf", accept_multiple_files=True)
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)
    st.markdown("---")
   
    st.markdown("---")
    
# Custom CSS for dark and light mode
if dark_mode:
    st.markdown("""
        <style>
        body, .main { background-color: #1e1e1e; color: #ffffff; }
        .stButton button, .stDownloadButton button, .stFileUploader input[type="file"] {
            background-color: #333333; color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main { background-color: #f0f0f5; padding: 20px; }
        .stButton button { background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px; }
        .stTextInput input { background-color: #ffffff; border-radius: 5px; padding: 10px; }
        .stTextArea textarea { background-color: #f8f8f8; border-radius: 5px; padding: 10px; }
        .stDownloadButton button { background-color: #2196F3; color: white; font-size: 14px; padding: 10px 20px; border-radius: 5px; }
        .stFileUploader input[type="file"] { background-color: #2196F3; color: white; font-size: 14px; padding: 10px 20px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

# Session states
if "history" not in st.session_state:
    st.session_state.history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Handle PDF processing
if pdf_files:
    with st.spinner("Processing PDFs..."):
        text = extract_text_from_pdfs(pdf_files)
        chunks = chunk_text(text)
        vector_store = create_vector_store(chunks)
        st.session_state.chunks = chunks
        st.success("PDFs processed successfully!")

# User input for chat
user_question = st.text_input("Ask a question to the document 🤔")

# Answering the question
if user_question:
    context = "\n".join([chunk[:500] for chunk in st.session_state.chunks[:3]])
    prompt = f"Context: {context}\n\nQuestion: {user_question}"
    answer = chat_with_groq(prompt)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({"question": user_question, "answer": answer, "timestamp": timestamp})

    for chat in st.session_state.history:
        st.markdown(f"**User ({chat['timestamp']}):** {chat['question']} 🧑‍💻")
        st.markdown(f"**Groq Model:** {chat['answer']} 🤖")
        st.write("---")

# Download history
if st.session_state.history:
    history_str = "\n".join([f"Q: {chat['question']}\nA: {chat['answer']}\nTimestamp: {chat['timestamp']}\n" for chat in st.session_state.history])
    st.download_button("Download Chat History 📥", data=history_str, file_name="chat_history.txt", mime="text/plain")

    history_json = json.dumps(st.session_state.history, indent=4)
    st.download_button("Download Chat History (JSON) 📥", data=history_json, file_name="chat_history.json", mime="application/json")

# PDF summary
if st.session_state.chunks:
    if st.button("Generate PDF Summary 📝"):
        summary = summarize_text(" ".join(st.session_state.chunks))
        st.write("**Summary:**")

        st.write(summary)




