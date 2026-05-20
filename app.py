# ============================================================
# RAG PDF QUESTION ANSWERING CHATBOT USING GROQ + LANGCHAIN
# FILE NAME: app.py
# ============================================================

import os
import shutil

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings
)

from langchain.prompts import PromptTemplate

# ============================================================
# STEP 1: SET GROQ API KEY
# ============================================================

GROQ_API_KEY = "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"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ============================================================
# STEP 2: PDF FILE SELECTION
# ============================================================

print("\n======================================")
print("      PDF DOCUMENT SELECTION")
print("======================================")

pdf_path = input("\nEnter PDF File Path: ")

# ============================================================
# VALIDATE FILE
# ============================================================

if not os.path.exists(pdf_path):

    print("\n❌ Error: PDF file not found.")
    exit()

if not pdf_path.endswith(".pdf"):

    print("\n❌ Error: Please provide a PDF file.")
    exit()

print("\n✅ PDF File Found Successfully!")

# ============================================================
# STEP 3: LOAD PDF DOCUMENT
# ============================================================

try:

    print("\n📄 Loading PDF Document...")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"\n✅ Total Pages Loaded: {len(documents)}")

except Exception as e:

    print("\n❌ Error Loading PDF:")
    print(str(e))
    exit()

# ============================================================
# STEP 4: SPLIT DOCUMENT INTO CHUNKS
# ============================================================

try:

    print("\n✂️ Splitting Document into Chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    print(f"\n✅ Total Chunks Created: {len(docs)}")

except Exception as e:

    print("\n❌ Error While Splitting Text:")
    print(str(e))
    exit()

# ============================================================
# STEP 5: LOAD EMBEDDING MODEL
# ============================================================

try:

    print("\n🧠 Loading Embedding Model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n✅ Embedding Model Loaded Successfully!")

except Exception as e:

    print("\n❌ Error Loading Embedding Model:")
    print(str(e))
    exit()

# ============================================================
# STEP 6: REMOVE OLD VECTOR DATABASE
# ============================================================

if os.path.exists("./chroma_db"):

    shutil.rmtree("./chroma_db")

# ============================================================
# STEP 7: CREATE VECTOR DATABASE
# ============================================================

try:

    print("\n🗂️ Creating Vector Database...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("\n✅ Vector Database Created Successfully!")

except Exception as e:

    print("\n❌ Error Creating Vector Database:")
    print(str(e))
    exit()

# ============================================================
# STEP 8: CREATE RETRIEVER
# ============================================================

try:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    print("\n✅ Retriever Created Successfully!")

except Exception as e:

    print("\n❌ Error Creating Retriever:")
    print(str(e))
    exit()

# ============================================================
# STEP 9: LOAD GROQ LLM
# ============================================================

try:

    print("\n🤖 Loading GROQ LLM Model...")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    print("\n✅ GROQ LLM Loaded Successfully!")

except Exception as e:

    print("\n❌ Error Loading GROQ Model:")
    print(str(e))
    exit()

# ============================================================
# STEP 10: CREATE CUSTOM PROMPT
# ============================================================

custom_prompt = PromptTemplate(
    template="""
Use the following pieces of context to answer the user's question.

Rules:
1. Answer only from the provided context
2. Do not hallucinate
3. If answer is unavailable, say:
   "I don't know based on the provided document."
4. Keep answers clear and professional
5. Always end with:
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
# STEP 11: CREATE RAG QA CHAIN
# ============================================================

try:

    print("\n🔗 Creating Retrieval QA Chain...")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": custom_prompt
        }
    )

    print("\n✅ RAG QA Chain Created Successfully!")

except Exception as e:

    print("\n❌ Error Creating QA Chain:")
    print(str(e))
    exit()

# ============================================================
# STEP 12: START CHATBOT
# ============================================================

print("\n======================================")
print("        SIMPLE RAG PDF CHATBOT")
print("======================================")

while True:

    question = input("\nAsk Question (type 'exit' to quit): ")

    if question.lower() == "exit":

        print("\n👋 Exiting RAG Chatbot...")
        break

    if question.strip() == "":

        print("\n⚠️ Please enter a valid question.")
        continue

    try:

        response = qa_chain.invoke({
            "query": question
        })

        print("\n======================================")
        print("ANSWER")
        print("======================================\n")

        print(response["result"])

    except Exception as e:

        print("\n❌ Error Occurred:")
        print(str(e))
