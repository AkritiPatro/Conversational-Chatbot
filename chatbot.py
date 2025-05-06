# 🚀 Import necessary libraries
import streamlit as st
import os
from langchain_groq import ChatGroq
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from dotenv import load_dotenv

# 🌍 Load environment variables from .env file
load_dotenv()

# 🔐 API Keys from environment
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# ❌ Check for missing API keys
if not groq_api_key or not google_api_key:
    st.error("❌ ERROR: Missing API keys! Please check your .env file.")
    st.stop()

# 🧠 Define available LLM models
model_options = {
    "Llama3-8B (Groq)": "llama3-8b-8192",
    "Mixtral (Groq)": "mixtral-8x7b-32768",
    "Gemini Pro (Google)": "gemini-pro"
}

# 📂 Sidebar for managing chats
st.sidebar.title("📂 Chats")
if "chats" not in st.session_state:
    st.session_state.chats = {}

# 🧠 Model selection stored in session
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# ➕ Create new chat
if st.sidebar.button("➕ New Chat"):
    chat_name = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_name] = []
    st.session_state.current_chat = chat_name
    st.session_state.file_uploaded = None
    st.session_state.selected_model = None
    st.rerun()

# ❌ Delete specific chat
for chat_name in list(st.session_state.chats.keys()):
    cols = st.sidebar.columns([0.8, 0.2])
    if cols[0].button(f"📝 {chat_name}", key=chat_name):
        st.session_state.current_chat = chat_name
        st.rerun()
    if cols[1].button("❌", key=f"delete_{chat_name}"):
        del st.session_state.chats[chat_name]
        st.session_state.current_chat = list(st.session_state.chats.keys())[0] if st.session_state.chats else None
        st.session_state.file_uploaded = None
        st.session_state.selected_model = None
        st.rerun()

# 🗑️ Delete all chats
if st.sidebar.button("🗑️ Delete All Chats"):
    st.session_state.chats = {}
    st.session_state.current_chat = None
    st.session_state.file_uploaded = None
    st.session_state.selected_model = None
    st.rerun()

# 🛡️ Ensure at least one chat exists
if "current_chat" not in st.session_state or not st.session_state.current_chat:
    if st.session_state.chats:
        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
    else:
        chat_name = f"Chat 1"
        st.session_state.chats[chat_name] = []
        st.session_state.current_chat = chat_name

# 🏠 Main Chatbot UI
st.title("💬 Conversational Chatbot")
selected_model = st.selectbox("🧠 Choose an LLM Model:", list(model_options.keys()))

# 📤 Upload file
uploaded_file = st.file_uploader("📂 Upload a document (PDF, TXT, DOCX, XLSX):", type=["pdf", "txt", "docx", "xlsx"])

# 🔄 File persistence in session state
if uploaded_file:
    st.session_state.file_uploaded = uploaded_file
    st.success("✅ File uploaded successfully!")
elif "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
    file_content = ""
else:
    uploaded_file = st.session_state.file_uploaded

# 📄 Function to extract text from uploaded files
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file, sheet_name=None)
        text = []
        for sheet, data in df.items():
            text.append(f"📄 Sheet: {sheet}\n")
            text.append(data.to_string(index=False))
        return "\n\n".join(text)
    return ""

file_content = extract_text(uploaded_file) if uploaded_file else ""

# 🤖 Set up the selected LLM model
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model

if selected_model.startswith("Gemini"):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("gemini-pro")
else:
    model = ChatGroq(groq_api_key=groq_api_key, model_name=model_options[selected_model])

# 💬 Chat interface
st.subheader("💬 Chat")

# 📜 Show chat history
if st.session_state.current_chat in st.session_state.chats:
    for chat in st.session_state.chats[st.session_state.current_chat]:
        if chat["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(chat["text"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(chat["text"])

# 🔡 User input for questions
user_input = st.chat_input("💡 Ask a question based on the file...")

# ⚙️ Handle input and generate response
if user_input and file_content:
    st.session_state.chats[st.session_state.current_chat].append({"role": "user", "text": user_input})

    prompt = f"Here's the document content:\n\n{file_content}\n\nBased on this, answer the question: {user_input}"

    if selected_model.startswith("Gemini"):
        response = model.generate_content(prompt)
        answer = response.text
    else:
        response = model.invoke(prompt)
        answer = response.content if hasattr(response, "content") else "No answer found."

    st.session_state.chats[st.session_state.current_chat].append({"role": "bot", "text": answer})
    st.rerun()

elif user_input:
    st.warning("⚠️ Please upload a file before asking a question.")
