import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def process_text(text): 
    # CharacterTextSplitter로 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Ollama 임베딩 생성
    embeddings = OllamaEmbeddings(model="exaone3.5")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():
    st.title("📄PDF 요약하기 (Ollama 기반)")
    st.divider()

    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."

        if query:
            docs = documents.similarity_search(query)

            # Ollama 기반 LLM 사용 (예: llama3, mistral 등)
            llm = ChatOllama(model="EEVE-Korean-10.8B", temperature=0.1)
            prompt = PromptTemplate.from_template(
                "다음 문서를 참고해서 질문에 답해주세요:\n\n{context}\n\n질문: {question}"
            )
            chain = create_stuff_documents_chain(llm, prompt)
            response = chain.invoke({"context": docs, "question": query})

            st.subheader('--요약 결과--:')
            st.write(response)

if __name__ == '__main__':
    main()