# RAG-based-LLM-Application

Overview
This project is a Retrieval-Augmented Generation (RAG) application designed to provide context-aware answers using a Large Language Model (LLM). By leveraging a vector database (e.g., Chroma, Pinecone), the system stores word embeddings of custom documents. When a user submits a query, the retriever fetches the top k similar context embeddings from the vector database. These embeddings are then used to provide the LLM with relevant context, enabling it to generate accurate and contextually appropriate responses based on recent and custom company data

# Prerequisites
Python 3.8 or higher
pip (Python package installer)
Access to a vector database (e.g., Chroma, Pinecone)
API keys or credentials for the vector database and LLM

# Tools Used
- LangChain
- Python   
- FAISS
- OpenAI
- Streamlit
