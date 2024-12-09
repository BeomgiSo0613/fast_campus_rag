## VSCode 설정

- VSCode 사용법 설정에 대한 설명입니다.

# Chapter 1: 프롬프트

### 1. PromptTemplate, 부분 변수 (partial_variables)

```python
from dotenv import load_dotenv

load_dotenv()
```

- API 연결 확인

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH02-Prompt")
```

- LangSmith 연결 설정

---

### Prompt Method 사용

#### 1.1 `from_template` 이용

```python
from langchain_core.prompts import PromptTemplate

# 템플릿 정의. {country}는 변수로, 이후에 값이 들어갈 자리를 의미
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)

# prompt 생성. format 메소드를 이용하여 변수에 값을 넣어줌
prompt = prompt.format(country="대한민국")

# chain 생성
chain = prompt | llm

# country 변수에 입력된 값이 자동으로 치환되어 수행됨
# invoke 변수가 여러 개일 때는 dict 형태로 넣어주어야 합니다.
chain.invoke({"country": "대한민국"}).content
```
- 출력: '대한민국의 수도는 서울입니다.'


#### 1.2 PromptTemplate 객체 생성과 동시에 프롬프트 생성

```python
# 템플릿 정의
template = "{country}의 수도는 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)

prompt
```
- `PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='{country}의 수도는 어디인가요?')`

- 객체에서 필요한 값을 넣어줍니다. `from_template`과 PromptTemplate의 차이점입니다.

```python
# prompt 생성
prompt.format(country="대한민국")

# 템플릿 정의
template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "미국"  # dictionary 형태로 partial_variables를 전달
    },
)

prompt
```

```python
prompt.format(country1="대한민국")

prompt = PromptTemplate.from_template(template)
prompt

# partial을 통해 부분 변수를 입력할 수 있습니다.
prompt_partial = prompt.partial(country2="캐나다")
prompt_partial

# chain으로 묶습니다.
chain = prompt_partial | llm

chain.invoke("대한민국").content

# 덮어씁니다.
chain.invoke({"country1": "대한민국", "country2": "호주"}).content
```

- `partial_variables`: 부분 변수 채움

`partial`을 사용하는 일반적인 용도는 함수를 부분적으로 사용하는 것입니다. 이 사용 사례는 **항상 공통된 방식으로 가져오고 싶은 변수**가 있는 경우입니다.

대표적인 예가 **날짜나 시간**입니다.

항상 현재 날짜가 표시되기를 원하는 프롬프트가 있다고 가정해 보겠습니다. 프롬프트에 하드코딩할 수도 없고, 다른 입력 변수와 함께 전달하는 것도 번거롭습니다. 이 경우 항상 현재 **날짜를 반환하는 함수**를 사용하여 프롬프트를 부분적으로 변경할 수 있으면 매우 편리합니다.

### 2. YAML 형식으로 프롬프트 읽기

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml", encoding="utf-8")
prompt

from langchain_teddynote.prompts import load_prompt

# Windows 사용자 only: 인코딩을 cp949로 설정
load_prompt("prompts/fruit_color.yaml", encoding="utf-8")

prompt.format(fruit="사과")

prompt2 = load_prompt("prompts/capital.yaml")
print(prompt2.format(country="대한민국"))

from langchain_core.output_parsers import StrOutputParser
chain = prompt2 | llm

# topic을 invoke에 넣어야 한다고 생각!
chain = prompt2 | llm | StrOutputParser()

answer = chain.invoke({"country": "대한민국"})

answer
```

- 출력 예시:
  '1. 면적 - 서울 특별시 면적은 약 605.21km²로 한반도 중앙에 위치하고 있다.\n\n2. 인구 - 서울 특별시의 인구는 약 9,700만 명으로 대한민국 인구의 약 1/5을 차지하고 있다.\n\n3. 역사적 장소 - 경복궁, 창경궁, 덕수궁 등 다양한 조선 시대의 왕궁과 궁전이 위치해 있으며, 종묘, 선릉 등 역사적인 유적지도 많이 존재한다.\n\n4. 특산품 - 한복, 한지, 불고기, 삼겹살, 떡볶이, 김치 등 다양한 대한민국 전통 음식과 문화 상품들이 수도인 서울에서 만날 수 있다.'

### 3. ChatPromptTemplate의 from_messages
- 대화형 형식으로 대화하기 좋은 템플릿

- `ChatPromptTemplate`은 대화 목록을 프롬프트로 주입하고자 할 때 활용할 수 있습니다.

- 메시지는 튜플(tuple) 형식으로 구성하며, (`role`, `message`)로 구성하여 리스트로 생성할 수 있습니다.

#### 역할 설명

- `"system"`: 시스템 설정 메시지입니다. 주로 전역 설정과 관련된 프롬프트입니다.
  - AI의 역할, 대화가 시작되고 끝날 때까지의 환경 설정
- `"human"`: 사용자 입력 메시지입니다.
  - 우리가 입력한 질문 값
- `"ai"`: AI의 답변 메시지입니다.

```python
ChatPromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='{country}의 수도는 어디인가요?'), additional_kwargs={})])
```

- messages라는 값이 생성됩니다.

- Human
- (role, message)
- message: AI 답변, 역할, 전역 prompt이다.

- 생성한 메시지를 바로 주입하여 결과를 받을 수 있습니다.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message 메시지는 튜플 형식으로 넣습니다.
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

# 챗 message를 생성합니다.
messages = chat_template.format_messages(
    name="테디", user_input="당신의 이름은 무엇입니까?"
)
messages

llm.invoke(messages).content

chain = chat_template | llm

chain.invoke({"name": "Teddy", "user_input": "당신의 이름은 무엇입니까?"}).content
```

### 3.2 MessagePlaceholder

- 또한 LangChain은 포맷하는 동안 렌더링할 메시지를 완전히 제어할 수 있는 `MessagesPlaceholder`를 제공합니다.

- 메시지 프롬프트 템플릿에 어떤 역할을 사용해야 할지 확실하지 않거나 서식 지정 중에 메시지 목록을 삽입하려는 경우 유용할 수 있습니다.

- `conversation` 대화 목록을 나중에 추가하고자 할 때 `MessagesPlaceholder`를 사용할 수 있습니다.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
    ]
)
chat_prompt

formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)

print(formatted_chat_prompt)

# chain 생성
chain = chat_prompt | llm | StrOutputParser()

# chain 실행 및 결과 확인
chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다.",
            ),
            ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
        ],
    }
)

```

### 5. Few shot prompt

- 