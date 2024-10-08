import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from langchain_community.llms import HuggingFaceHub
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import api

# Initialize global variables
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
history = []
db = None

def create_prompt_template():
    template = """
    You are a strict and accurate AI assistant. Your task is to answer questions based ONLY on the given context. Follow these rules:

    1. If the question contains spelling or grammatical mistakes, correct them and include both the original and corrected versions in your answer.
    2. If the question is not related to the given context, respond with EXACTLY: "The question is not related to the given context."
    3. If you don't have enough information to answer the question based on the context, respond with EXACTLY: "I don't have enough information to answer that question based on the given context."
    4. Do not use any external knowledge or make assumptions beyond what's provided in the context.
    5. Provide concise, factual answers based solely on the information in the context.
    6. If you're unsure about any part of the answer, state that clearly.
    7. The question will be asked thinking that you are that person in the context.
    8. Add punctuations in question if required.

    Context: {context}

    Question: {question}

    Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=template)

# def store_history(history, query, response):
#     history.append(HumanMessage(content=query))
#     history.append(AIMessage(content=response))
#     return history

def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large", 
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=api.huggingfacehub_api_token
    )

def load_docx(file_path):
    """Load content from a .docx file."""
    doc = Document(file_path)
    text = [para.text for para in doc.paragraphs]
    return "\n".join(text)

def load_or_create_db(file_path):
    global db
    if not os.path.exists("chroma_db"):
        document_text = load_docx(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(document_text)
        db = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")
        db.persist()
    else:
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return db

def clean_response(response):
    # Split the response by "AI:" and take the last part
    parts = response.split("AI:")
    if len(parts) > 1:
        return parts[-1].strip()
    return response.strip()

# Streamlit app
st.title("Arivazhagan's Portfolio")

# Load database and initialize LLM
db = load_or_create_db("Documented Resume.docx")
llm = get_llm()
prompt_template = create_prompt_template()

# User input
user_query = st.text_input("Enter your question:")

if user_query:
    # Perform similarity search
    results = db.similarity_search(user_query, k=3)
    context = " ".join([doc.page_content for doc in results])
    print(context)
    # history_context = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history])
    
    prompt = prompt_template.format(context=context, question=user_query)
    # prompt = create_prompt_template(context, user_query)
    
    response = llm(prompt)
    
    st.write(f"Answer: {response}")
    
    # history = store_history(history, user_query, response)
    # print(history)