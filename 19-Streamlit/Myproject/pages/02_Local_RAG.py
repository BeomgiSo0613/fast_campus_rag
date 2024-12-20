import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_teddynote.prompts import load_prompt
import glob
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from retriever import create_retriever
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv



# 파일 업로드 전용 폴더
# 캐시 디렉토리 생성
if not os.path.exists("cache"):
    os.mkdir("cache")
    
if not os.path.exists("cache/files"):
    os.mkdir("cache/files")

# API 키 정보 로드
load_dotenv()

logging.langsmith("PDF-RAG-STREAMLIT")
st.title("Local 모델 기반 RAG")


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "chain" not in st.session_state:
    # 아무런 파일을 업로드하지 않을 경우
    st.session_state["chain"] = None   
    
    
with st.sidebar:
    # 대화 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    
    # 파일 업로더
    uploaded_file = st.file_uploader("파일 업로드", type = ["pdf"])
    
    selected_model=st.selectbox("LLM 선택", ["gpt-4o",'gpt-4-turbo','gpt-4o-mini'],index = 0)

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

@st.cache_resource(show_spinner = "업로드한 파일을 처리중입니다....")
def embed_file(file):
    # 업로드한 파일을 캐시 디랙토리에 저장합니다.
    file_content = file.read()
    file_path = f"./cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    retriever = create_retriever(file_path)
    
    return retriever
def create_chain(retriever, model_name = 'gpt-4o'):
    
    
    prompt = load_prompt("prompts/pdf_rag.yaml", encoding = "utf-8")
    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} # retriever 필요한 document를 가져온다. RunnablePassthrough 질문한 내용이 그대로간다.
        | prompt
        | llm
        | StrOutputParser()
    )
    # 체인 생성
    
    
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성(작업시간이 오래걸릴 예정...)
    retriever = embed_file(uploaded_file)

    chain = create_chain(retriever,model_name = selected_model)

    st.session_state["chain"] = chain

    
print_messages()

user_input = st.chat_input("궁금한게 무엇인가요?")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 사용자의 입력 처리
if user_input:

    # chain 설정
    # 적절한 프롬프트를 넣어주어야한다.
    chain = st.session_state["chain"]
    
    if chain is not None:
        # 웹의 사용자 정의
        st.chat_message("user").write(user_input)

        # 스트리밍
        # 답변을 조금더 빠르게 하려면
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈공간을 만들어, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            
            ai_answer = ""
            for token in response:        
                ai_answer += token
                container.markdown(ai_answer)
        # AI 답변 출력
    #    st.chat_message("assistant").write(ai_answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
        
    else:
        warning_msg.error("파일을업로드해주세요")