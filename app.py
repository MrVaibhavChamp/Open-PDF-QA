import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInferenceAPIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    embeddings = HuggingFaceInferenceAPIEmbeddings(model_name="WhereIsAI/UAE-Large-V1", api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
    return vectorstore

def get_conv_chain(vectorstore):
    # llm = HuggingFaceEndpoint(endpoint_url="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})   # With langchain_huggingface
    llm = HuggingFaceEndpoint(endpoint_url="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=512)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with multiple pdfs :books:")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click the 'Process' button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.conversation = get_conv_chain(vectorstore)
    
if __name__ == "__main__":
    main()