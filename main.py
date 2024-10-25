import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Ed-Tech FAQ Chatbot")

btn = st.button("Create Knowledge Base")

if btn:
    pass

question = st.text_input("Enter your question")


if question:
    chain = get_qa_chain()
    result = chain.invoke({"query": question})
    st.header("Answer:")
    st.write(result["result"])