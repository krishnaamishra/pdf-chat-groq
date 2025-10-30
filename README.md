📚 PDF Chat with Groq LLaMA 3.3 🤖

This project is a Streamlit web app that lets you upload PDF documents and chat with them using Groq’s LLaMA 3.3 large language model.
It automatically extracts text from your PDFs, creates semantic embeddings for intelligent search, and generates accurate, context-aware answers to your questions.

🚀 Features

📂 Upload multiple PDFs — works with any readable document

🧠 Chat with your documents — ask natural language questions

✂️ Smart text chunking — splits long documents for better understanding

💬 Conversational memory — keeps full chat history

🧾 Automatic summarization — get concise summaries of large PDFs

🌙 Dark/Light mode UI — modern, responsive design

💾 Download chat history — export your Q&A as .txt or .json

⚙️ Tech Stack
Technology	Purpose
Streamlit	Front-end web app interface
Groq API (LLaMA 3.3 model)	Natural language understanding & response generation
LangChain	Text splitting and vector database logic
Hugging Face Embeddings	Semantic text embeddings
ChromaDB	Local vector storage for efficient similarity search
PyPDF2	PDF text extraction
PIL / pytesseract	Optional OCR support for image-based PDFs
🧩 How It Works

Upload PDFs — You upload one or more PDF files.

Text Extraction — Text is extracted from each page.

Text Chunking — The document is split into smaller overlapping chunks.

Vector Embedding — Each chunk is converted into a numerical vector using a SentenceTransformer model.

Context Retrieval — When you ask a question, the app retrieves relevant chunks.

Groq LLaMA Query — It sends context + your question to Groq’s LLaMA model for a precise answer.

Response Display — The model’s answer appears instantly in your chat window.

🧾 Deployment Options

Local:

streamlit run app.py


Cloud (Recommended):
Deploy easily on Streamlit Cloud
 by connecting your GitHub repo.

🔐 API Key Setup

Add your Groq API Key in Streamlit Secrets:

GROQ_API_KEY = "your_actual_groq_api_key_here"


Access it in your code:

import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

🧠 Future Improvements

Add PDF content search and filtering

Support scanned PDFs with OCR integration

Save persistent chat sessions

Export summaries as PDF reports

🧑‍💻 Author

Krishna Mishra
Built with ❤️ using Python, Streamlit, and Groq LLaMA 3.3.
