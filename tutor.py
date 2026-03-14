import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

st.title("AI Tutor - Anytime Anywhere  ")

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:

    docs = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3.2")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask your question")

    if query:

        result = qa.invoke({"query": query})

        st.write("### Answer")
        st.write(result["result"])

 