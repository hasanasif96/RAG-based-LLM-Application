import json
import sys
from pypdf import PdfReader
import streamlit as st
import numpy as np
import os
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfFileReader

os.environ["OPENAI_API_KEY"] = ""


## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)

    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        OpenAIEmbeddings()
    )
    vectorstore_faiss.save_local("faiss_index")


def get_llm():
    os.environ["OPENAI_API_KEY"] = ""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    return llm


prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {question}""")


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa({"query": query})
    return answer["result"]


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using RAG")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",OpenAIEmbeddings())
            llm = get_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")




if __name__ == "__main__":
    main()













