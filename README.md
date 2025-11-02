# 📚 PDF Chat with Groq Llama3.1 🤖

This Streamlit web app allows users to upload one or more PDF files, extract their text, and interact with the content using Groq's Llama3.1 model.

### 🚀 Features
- Upload multiple PDFs
- Extract text and chunk it for better processing
- Ask contextual questions about the document
- Generate document summaries
- Download full chat history (TXT or JSON)
- Light/Dark mode interface

### 🧠 Tech Stack
- Streamlit (Frontend)
- Groq API (LLM backend)
- LangChain + HuggingFace embeddings
- ChromaDB (Vector store)
- PyPDF2 / OCR (PDF parsing)

### 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
