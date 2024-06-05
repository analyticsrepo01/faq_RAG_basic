import streamlit as st
import pandas as pd
import socket
import re
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai

# Helper function to create vector DB
def creation_of_vectorDB_in_local(data, embeddings):
    db = FAISS.from_texts(data, embeddings)
    db.save_local("FAISS_Index")

# Helper function to create the FAQ chain
def creation_FAQ_chain():
    db = FAISS.load_local("FAISS_Index", embeddings)
    retriever = db.as_retriever(score_threshold=0.7)
    llm = VertexAI(model_name="gemini-pro")
    prompt_temp = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "This Question not Present in My Database." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""
    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, input_key="query", return_source_documents=False, chain_type_kwargs={"prompt": PROMPT})
    return chain

# Streamlit application starts here
st.title("CSV to VertexAI FAQ App")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded CSV file:")
    st.dataframe(df)

    # Extract data from CSV
    data = df["question"].tolist()
    
    # Initialize VertexAI embeddings
    model_name = 'textembedding-gecko@latest'
    embeddings = VertexAIEmbeddings(model_name)
    
    # Create vector DB
    creation_of_vectorDB_in_local(data, embeddings)
    
    # Create FAQ chain
    chain = creation_FAQ_chain()
    
    # Input prompt
    prompt = st.text_input("Enter your question:")
    
    if st.button("Get Response"):
        if prompt:
            result = chain({"query": prompt})
            st.write("Response:", result["result"])
        else:
            st.write("Please enter a question.")
