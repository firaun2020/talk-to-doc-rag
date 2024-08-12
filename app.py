import streamlit as stl
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
   # llm = OpenAI()
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",temperature=0.1, max_length=128)
   # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm (
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversational_chain

def handle_user_input(user_question):
    response = stl.session_state.conversation({"question": user_question})
    stl.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(reversed(stl.session_state.chat_history)):
        if i % 2 == 0:
           stl.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            stl.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            

def main():
    load_dotenv()
    stl.set_page_config(page_title="My RAG Attempt!", page_icon=":robot:")

    stl.write(css, unsafe_allow_html=True)

    if "conversation" not in stl.session_state:
        stl.session_state.conversation = None

    if "chat_history" not in stl.session_state:
        stl.session_state.chat_history = None

    stl.header("My RAG Attempt! :robot:")



    user_question = stl.text_input("Start the interview: ", value="", key="uq")
    if user_question:
        handle_user_input(user_question)


    # stl.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    # stl.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with stl.sidebar:
        stl.subheader("Upload CV")
        pdf_doc = stl.file_uploader("Upload your CV here", accept_multiple_files=True)
        if stl.button("Submit"):
            with stl.spinner("Processing..."):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_doc)
                #stl.write(raw_text)

                # Get the chunks of text
                text_chunks = get_text_chunks(raw_text)
               # stl.write(text_chunks)

                # Create Vector store
                vector_store = get_vectorstore(text_chunks)

                # Conversation chain
                stl.session_state.conversation = get_conversation_chain(vector_store)

                stl.write("Processing Completed")


   


if __name__ == '__main__':
    main()