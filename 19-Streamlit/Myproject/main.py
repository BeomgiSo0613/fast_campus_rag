# import streamlit as st

# st.title("Chat GPT 활용하기")


# # st.markdown("*Streamlit* is **really** ***cool***.")
# # st.markdown('''
# #     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
# #     :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
# # st.markdown("Here's a bouquet &mdash;\
# #             :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

# # multi = '''If you end a line with two spaces,
# # a soft return is used for the next line.

# # Two (or more) newline characters in a row will result in a hard return.
# # '''
# # st.markdown(multi)

# # 채팅을 입력할 창이 필요

# user_input = st.chat_input("궁금한게 무엇인가요?")
# # 사용자의 입력이 들어오묜
# if user_input:
#     #st.write(f"사용자입력: {user_input}")
    
#     #st.chat_message("user").write(user_input)
#     # 위의 구문과 동일
#     # with st.chat_message("user"):
#         #st.write(user_input)
    
#     # ai bot만들기
#     st.chat_message("user").write(user_input)
#     st.chat_message("assistance").write(user_input)
        
        
        
# st.chat_msaage -> container역할

## 입력하면 다시 처음부터 매번 새로고침이 일어남
## session_state가 필요함

#######
import streamlit as st
from langchain_core.messages.chat import ChatMessage

st.title("Chat GPT 활용하기")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
# 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []


# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)

# 이전대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
 #       st.write(f"{chat_message.role} : {chat_message.content}")
        st.chat_message(chat_message.role).write(chat_message.content)
        
# 
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role = role, content = message))
# 채팅을 입력할 창이 필요
user_input = st.chat_input("궁금한게 무엇인가요?")


print_messages()


# 사용자의 입력이 들어오면
if user_input:    
    # 웹에 대화를 출력
    st.chat_message("user").write(user_input)
    st.chat_message("assistance").write(user_input)
    
    # 대화기록을 저장한다.
    # ChatMessage(role="user", content=user_input)
    # ChatMessage(role="assistance", content=user_input)
    
    # st.session_state["messages"].append(("user",user_input))
    # st.session_state["messages"].append(("assistance",user_input))
    # 함수로 위의 역할을 명확하게 구분
    add_message("user",user_input)
    add_message("assistant", user_input)