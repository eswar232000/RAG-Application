# ============================================================
# RAG PDF QUESTION ANSWERING CHATBOT USING GROQ + LANGCHAIN
# FILE NAME: app.py
# ============================================================

import os

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# ============================================================
# STEP 1: SET GROQ API KEY
# ============================================================

GROQ_API_KEY = "gsk_6TkIhMnwZJpJFU0uqiG5WGdyb3FYfGgLojgSjOzgn8rmjGxiEvdAY"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ============================================================
# STEP 2: GET PDF FILE PATH FROM USER
# ============================================================

print("\n======================================")
print("      PDF DOCUMENT SELECTION")
print("======================================")

pdf_path = input("\nEnter PDF File Path: ")

# Check if file exists
if not os.path.exists(pdf_path):

    print("\nError: PDF file not found.")
    exit()

# ============================================================
# STEP 3: LOAD PDF DOCUMENT
# ============================================================

print("\nLoading PDF Document...")

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"\nTotal Pages Loaded: {len(documents)}")

# ============================================================
# STEP 4: SPLIT DOCUMENT INTO CHUNKS
# ============================================================

print("\nSplitting Document into Chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(docs)}")

# ============================================================
# STEP 5: LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding Model Loaded Successfully!")

# ============================================================
# STEP 6: CREATE VECTOR DATABASE
# ============================================================

print("\nCreating Vector Database...")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\nVector Database Created Successfully!")

# ============================================================
# STEP 7: CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("\nRetriever Created Successfully!")

# ============================================================
# STEP 8: LOAD GROQ LLM MODEL
# ============================================================

print("\nLoading GROQ LLM Model...")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("\nLLM Loaded Successfully!")

# ============================================================
# STEP 9: CREATE CUSTOM PROMPT
# ============================================================

custom_prompt = PromptTemplate(
    template="""
Use the following context to answer the user's question.

If you don't know the answer, say:
"I don't know based on the provided document."

Do not make up answers.

Always end the answer with:
"Thanks for asking!"

Context:
{context}

Question:
{question}

Helpful Answer:
""",
    input_variables=["context", "question"]
)

# ============================================================
# STEP 10: CREATE RAG QA CHAIN
# ============================================================

print("\nCreating Retrieval QA Chain...")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": custom_prompt
    }
)

print("\nRAG QA Chain Created Successfully!")

# ============================================================
# STEP 11: START CHATBOT
# ============================================================

print("\n======================================")
print("        SIMPLE RAG PDF CHATBOT")
print("======================================")

while True:

    question = input("\nAsk Question (type 'exit' to quit): ")

    if question.lower() == "exit":

        print("\nExiting RAG Chatbot...")
        break

    try:

        response = qa_chain.run(question)

        print("\n======================================")
        print("ANSWER")
        print("======================================\n")

        print(response)

    except Exception as e:

        print("\nError Occurred:")
        print(str(e))
