import streamlit as st
from langchain.chat_models import ChatOllama

# 페이지 설정
st.set_page_config(page_title="🦙 뭐든지 질문하세요~ ")
st.title('🦙 뭐든지 질문하세요~ ')

# 답변 생성 함수 (Ollama 사용)
def generate_response(input_text):
    llm = ChatOllama(
        model="llama3",  # 설치된 Ollama 모델명 (예: llama3, mistral, codellama 등)
        temperature=0,   # 창의성 설정
    )
    response = llm.predict(input_text)
    st.info(response)

# 질문 입력 UI
with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('보내기')
    if submitted:
        generate_response(text)