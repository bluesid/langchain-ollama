import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF에서 텍스트 추출
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# 텍스트를 청크로 분할
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 벡터 저장소 생성
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="exaone3.5")  # 적절한 임베딩 모델 선택
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# 대화 체인 구성
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="EEVE-Korean-10.8B"),  # 또는 "llama3" 등
        retriever=vectorstore.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h
    )
    return conversation_chain

# Streamlit UI
st.set_page_config(page_title="Ollama PDF QA Chatbot")
st.title("📄 PDF 챗봇 with Ollama")

user_uploads = st.file_uploader("📎 PDF 파일을 업로드해주세요", accept_multiple_files=True)

if user_uploads:
    if st.button("Upload"):
        with st.spinner("⏳ 문서를 처리 중입니다..."):
            raw_text = get_pdf_text(user_uploads)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

if user_query := st.chat_input("질문을 입력해주세요 💬"):
    if 'conversation' in st.session_state:
        result = st.session_state.conversation.invoke({
            "question": user_query,
            "chat_history": st.session_state.get("chat_history", [])
        })
        response = result["answer"]
    else:
        response = "📌 먼저 PDF 문서를 업로드해주세요."
    
    with st.chat_message("assistant"):
        st.write(response)