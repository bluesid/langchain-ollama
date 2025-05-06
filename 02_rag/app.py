from langchain_community.document_loaders import TextLoader
documents = TextLoader("AI.txt").load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서를 청크로 분할
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

docs = split_docs(documents)

# Ollama 임베딩 사용
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="exaone3.5")  # 다른 임베딩 모델로 교체 가능

# Chroma를 이용해 임베딩 저장
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings, persist_directory="rag_data")

# Ollama LLM 사용 (예: mistral, llama2 등)
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="EEVE-Korean-10.8B")  # 로컬에 설치된 모델 이름

# Q&A 체인 구성
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 질문하고 유사 문서 검색 후 답변
query = "AI란?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)