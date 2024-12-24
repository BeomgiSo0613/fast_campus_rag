import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain import hub
from langchain_teddynote.prompts import load_prompt


# API KEY 정보로드
load_dotenv()

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def create_chain(prompt_type):
    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            ("user", "#Question:\n{question}"),
        ]
    )    

    if prompt_type == "SNS 게시글":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
        
    elif prompt_type == "요약":
# set the LANGCHAIN_API_KEY environment variable (create key in settings)

        prompt = hub.pull("teddynote/chain-of-density-map-korean:ae651deb")
        
    ##

    # GPT
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    # 출력 파서
    output_parser = StrOutputParser()
    # 체인 생성
    chain = prompt | llm | output_parser
    
    
    return chain

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    # 대화 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    if clear_btn:
#        st.write("대화가 초기화되었습니다.")
        st.session_state["messages"] = []
        
    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요",
        ("기본모드","SNS 게시글","요약"),
        index = 0
    )

# 채팅 입력 창

st.title("Chat GPT 활용하기")
# 이전 대화 메시지 출력
print_messages()


user_input = st.chat_input("궁금한게 무엇인가요?")

# 사용자의 입력 처리
if user_input:
    # 웹의 사용자 정의
    st.chat_message("user").write(user_input)

    # chain 설정
    # 적절한 프롬프트를 넣어주어야한다.
    chain = create_chain(selected_prompt)


    # 스트리밍
    # 답변을 조금더 빠르게 하려면
    response = chain.stream({"question" :  user_input})
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
