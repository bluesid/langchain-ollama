import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import tempfile

uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    embeddings = OllamaEmbeddings(model="exaone3.5")  # 사용할 embedding 모델
    vectors = FAISS.from_documents(data, embeddings)

    # ChatOllama 설정
    llm = ChatOllama(model="EEVE-Korean-10.8B")  # 설치된 모델 이름에 맞게 수정 (예: "llama3", "mistral", "gemma" 등)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever()
    )

    def conversational_chat(query):
        result = chain({
            "question": query,
            "chat_history": st.session_state['history']
        })
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # 세션 상태 초기화
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["안녕하세요! " + uploaded_file.name + "에 대해 질문해주세요."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='Conv_Question', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="PDF 파일에 대해 질문해보세요 :)", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji", seed="Nala")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Fluffy")