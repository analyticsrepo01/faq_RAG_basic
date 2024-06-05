import streamlit as st
import pandas as pd
import vertexai
from vertexai.preview.generative_models import grounding
from vertexai.generative_models import GenerationConfig, GenerativeModel, Tool
from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers.bm25 import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Vertex AI Initialization ---
PROJECT_ID = "my-project-0004-346516"  # Replace with your project ID
vertexai.init(project=PROJECT_ID, location="us-central1")
data_store_path =  'projects/255766800726/locations/global/collections/default_collection/dataStores/singpost-pdf-per-page-qn_1717131756893' # Replace with your actual data_store_path
model = GenerativeModel(model_name="gemini-1.0-pro-002")
tool = Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(datastore=data_store_path)))
generation_config = GenerationConfig(temperature=0.1)

# --- Langchain Setup ---
embeddings = VertexAIEmbeddings('textembedding-gecko@latest')

# --- Streamlit App ---
st.title("Vertex AI Chatbot with CSV Knowledge Base")

# 1. Upload and Preview CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("CSV Preview")
    st.write(df.head())  # Show first few rows

    # 2. Create Vectorstore and Retriever
    data = df["question"].tolist()
    # bm25_retriever = BM25Retriever.from_texts(data, metadatas=[{"source": 1}] * len(data))
    # faiss_vectorstore = FAISS.from_texts(data, embeddings, metadatas=[{"source": 2}] * len(data))
    
    bm25_retriever = BM25Retriever.from_texts(data, metadatas=[{"source": i} for i in range(len(data))])
    faiss_vectorstore = FAISS.from_texts(data, embeddings, metadatas=[{"source": i} for i in range(len(data))])

    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

# 3. Get User Input
user_prompt = st.text_input("Enter your query:", "")
system_prompt = "You are a helpful FAQ bot for Singpost (Singppore postal services includeinf vpost) only provide answer from retrieved search results and do not add anything from your side , here are search results Q"

# 4. Process Input and Generate Response
if user_prompt:
    # Retrieve relevant documents
    docs = ensemble_retriever.invoke(user_prompt)
    # Convert document content into string
    st.write("docs",docs, docs[0])
    retrieved_result = ""
    first_result = 9999
    for doc in docs:
        if first_result == 9999:
            first_result = doc.metadata['source']
            retrieved_row = df.iloc[first_result] 
            retrieved_answer = retrieved_row['answer']  # Get the answer from the row
            retrieved_image_url = retrieved_row['image_references']  # Get URL from the row

            retrieved_result += f"Question : '{doc.page_content}'\nAnswer : '{retrieved_answer}'\n\nImage URL : '{retrieved_image_url}'\n\n"
    
    st.write(retrieved_result , "retrieved_row", retrieved_row)
    # Generate final response with the retrieved context
    response = model.generate_content(
        system_prompt + retrieved_result + user_prompt ,
        tools=[tool],
        generation_config=generation_config,
    )

    # 5. Display Response
    st.subheader("Response")
    # st.write(response)
    st.markdown(response.text)
    
    # image_url_from_response = response.text.split("Image Reference : '")[-1].split("'")[0]  # Extract image URL from response
    # if image_url_from_response:  # Check if image URL exists
    #     st.image(image_url_from_response)  # Display image if found