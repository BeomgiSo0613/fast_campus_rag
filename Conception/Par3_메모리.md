# 01. 메모리란?

## Why 메모리 사용해?
- ChatGPT : 이전에 있던 대화내용을 기억하려면? -> 메모리를 활용해야한다.
- Chathistory도 있지만, 분명 차이가 있다 추후 설명할 예정이다.

# 02. ConversationBufferMemory


- 이 메모리는 메시지를 저장한 다음 변수에 메시지를 추출할 수 있게 해줍니다.
- 먼저 문자열로 추출할 수 있습니다.
- 메모리에 가장 기본적인 메모리다
- 사람의 입력과 AI의 답변을 페어하게 만들어줘서 저장
    - 사실 메모리의 기본구조다!


```python
memory.save_context(
    inputs={
        "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
    },
    outputs={
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
    },
)
```

- 딕셔너리형식으로 한개의 턴을 저장

- memory 의 `load_memory_variables({})` 함수는 메시지 히스토리를 반환합니다.

- histroy라는 키로 저장

```python

# 'history' 키에 저장된 대화 기록을 확인합니다.
print(memory.load_memory_variables({})["history"])

```

- `save_context(inputs, outputs)` 메서드를 사용하여 대화 기록을 저장할 수 있습니다.

- 이 메서드는 `inputs`와 `outputs` 두 개의 인자를 받습니다.
- `inputs`는 사용자의 입력을, `outputs`는 AI의 출력을 저장합니다.
- 이 메서드를 사용하면 대화 기록이 `history` 키에 저장됩니다.
- 이후 `load_memory_variables` 메서드를 사용하여 저장된 대화 기록을 확인할 수 있습니다.


- 데이터 누적

```python
# 추가로 2개의 대화를 저장합니다.
memory.save_context(
    inputs={"human": "정보를 모두 입력했습니다. 다음 단계는 무엇인가요?"},
    outputs={
        "ai": "입력해 주신 정보를 확인했습니다. 계좌 개설 절차가 거의 끝났습니다. 마지막으로 이용 약관에 동의해 주시고, 계좌 개설을 최종 확인해 주세요."
    },
)
memory.save_context(
    inputs={"human": "모든 절차를 완료했습니다. 계좌가 개설된 건가요?"},
    outputs={
        "ai": "네, 계좌 개설이 완료되었습니다. 고객님의 계좌 번호와 관련 정보는 등록하신 이메일로 발송되었습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요. 감사합니다!"
    },
)

# 추가로 2개의 대화를 저장합니다.
memory.save_context(
    inputs={"human": "정보를 모두 입력했습니다. 다음 단계는 무엇인가요?"},
    outputs={
        "ai": "입력해 주신 정보를 확인했습니다. 계좌 개설 절차가 거의 끝났습니다. 마지막으로 이용 약관에 동의해 주시고, 계좌 개설을 최종 확인해 주세요."
    },
)
memory.save_context(
    inputs={"human": "모든 절차를 완료했습니다. 계좌가 개설된 건가요?"},
    outputs={
        "ai": "네, 계좌 개설이 완료되었습니다. 고객님의 계좌 번호와 관련 정보는 등록하신 이메일로 발송되었습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요. 감사합니다!"
    },
)

# history에 저장된 대화 기록을 확인합니다.
print(memory.load_memory_variables({})["history"])

```
- `return_messages=True` 로 설정하면 `HumanMessage` 와 `AIMessage` 객체를 반환합니다.

```python
memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    inputs={
        "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
    },
    outputs={
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
    },
)

memory.save_context(
    inputs={"human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"},
    outputs={
        "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
    },
)

memory.save_context(
    inputs={"human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"},
    outputs={
        "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
    },
)

# history에 저장된 대화 기록을 확인합니다.
memory.load_memory_variables({})["history"]

```

```
[HumanMessage(content='안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다.', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다.', additional_kwargs={}, response_metadata={})]
```
- 리스트 안에 humanMessage 객체 형식으로 들어가 있다.
- 좋은 이점?
    - 나중에 chain안에 바로 입력으로 들어갈 수 있다. 

```python
from langchain_core.prompts import ChatPromptTemplate

ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 봇입니다"),
        ("human", "대한민국의 수도는 어디야?"),
    ]
)

```

```
ChatPromptTemplate(input_variables=[], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='당신은 친절한 AI 봇입니다'), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='대한민국의 수도는 어디야?'), additional_kwargs={})])
```

## Chain 에 적용

1. `ConversationChain`을 사용하여 대화를 진행합니다.
2. 이전의 대화 기록을 기억하고 있는지 확인합니다.

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# LLM 모델을 생성합니다.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# ConversationChain을 생성합니다.
conversation = ConversationChain(
    # ConversationBufferMemory를 사용합니다.
    llm=llm,
    memory=ConversationBufferMemory(),
)


# 대화를 시작합니다.
response = conversation.predict(
    input="안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
)
print(response)

# 이전 대화내용을 불렛포인트로 정리해 달라는 요청을 보냅니다.
response = conversation.predict(
    input="이전 답변을 불렛포인트 형식으로 정리하여 알려주세요."
)
print(response)
```

- ConversationChain이 RunnableWithMessageHistory로 바꿔 추후 RunnableWithMessageHistory를 사용해야함.

- 단점 : 대화내용을 계속계속 저장해주면 나중에 너무 많은양의 대화를 저장하게 된다면, 입력토큰 이상의 대화내용을 말하게 된다.
    - 이로인해 API에 요청하는데 문제가 발생하기 된다.

---


# 02.ConversationBufferWindowMemory

- `ConversationBufferWindowMemory` 는 시간이 지남에 따라 대화의 상호작용 목록을 유지합니다.

- 이때, `ConversationBufferWindowMemory` 는 모든 대화내용을 활용하는 것이 아닌 **최근 K개** 의 상호작용만 사용합니다.

- 이는 버퍼가 너무 커지지 않도록 가장 최근 상호작용의 슬라이딩 창을 유지하는 데 유용할 수 있습니다.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True)

memory.save_context(
    inputs={
        "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
    },
    outputs={
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
    },
)
memory.save_context(
    inputs={"human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"},
    outputs={
        "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
    },
)
memory.save_context(
    inputs={"human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"},
    outputs={
        "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
    },
)
memory.save_context(
    inputs={"human": "인증번호를 입력했습니다. 계좌 개설은 이제 어떻게 하나요?"},
    outputs={
        "ai": "본인 인증이 완료되었습니다. 이제 원하시는 계좌 종류를 선택하고 필요한 정보를 입력해 주세요. 예금 종류, 통화 종류 등을 선택할 수 있습니다."
    },
)
memory.save_context(
    inputs={"human": "정보를 모두 입력했습니다. 다음 단계는 무엇인가요?"},
    outputs={
        "ai": "입력해 주신 정보를 확인했습니다. 계좌 개설 절차가 거의 끝났습니다. 마지막으로 이용 약관에 동의해 주시고, 계좌 개설을 최종 확인해 주세요."
    },
)
memory.save_context(
    inputs={"human": "모든 절차를 완료했습니다. 계좌가 개설된 건가요?"},
    outputs={
        "ai": "네, 계좌 개설이 완료되었습니다. 고객님의 계좌 번호와 관련 정보는 등록하신 이메일로 발송되었습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요. 감사합니다!"
    },
)

# 대화 기록을 확인합니다.
memory.load_memory_variables({})["history"]
```

# 03.ConversationTokenBufferMemory

- `ConversationTokenBufferMemory` 는 최근 대화의 히스토리를 버퍼를 메모리에 보관하고, 대화의 개수가 아닌 **토큰 길이** 를 사용하여 대화내용을 플러시(flush)할 시기를 결정합니다.

- `max_token_limit`: 대화 내용을 저장할 최대 토큰의 길이를 설정합니다.
- 실직적인 비용청구는 되지않음
```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI


# LLM 모델 생성
llm = ChatOpenAI(model_name="gpt-4o")

# 메모리 설정
memory = ConversationTokenBufferMemory(
    llm=llm, max_token_limit=150, return_messages=True  # 최대 토큰 길이를 50개로 제한
)

memory.save_context(
    inputs={
        "human": "안녕하세요, 저는 최근에 여러분 회사의 공작 기계를 구매했습니다. 설치 방법을 알려주실 수 있나요?"
    },
    outputs={
        "ai": "안녕하세요! 구매해 주셔서 감사합니다. 해당 기계 모델 번호를 알려주시겠어요?"
    },
)
memory.save_context(
    inputs={"human": "네, 모델 번호는 XG-200입니다."},
    outputs={
        "ai": "감사합니다. XG-200 모델의 설치 안내를 도와드리겠습니다. 먼저, 설치할 장소의 전원 공급 상태를 확인해주세요. 기계는 220V 전원이 필요합니다."
    },
)
memory.save_context(
    inputs={"human": "전원은 확인했습니다. 다음 단계는 무엇인가요?"},
    outputs={
        "ai": "좋습니다. 다음으로, 기계를 평평하고 안정된 바닥에 배치해 주세요. 이후, 제공된 사용자 매뉴얼에 따라 케이블 연결을 진행해 주시기 바랍니다."
    },
)
memory.save_context(
    inputs={"human": "연결은 어떻게 하나요?"},
    outputs={
        "ai": "매뉴얼의 5페이지를 참조해 주세요. 케이블 연결에 관한 상세한 지침이 있습니다. 이 과정에서 어려움이 있으시면 추가적으로 도와드리겠습니다."
    },
)
memory.save_context(
    inputs={"human": "설치가 완료되면 어떻게 해야 하나요?"},
    outputs={
        "ai": "설치가 완료되면, 전원을 켜고 초기 구동 테스트를 진행해 주시기 바랍니다. 테스트 절차는 매뉴얼의 10페이지에 설명되어 있습니다. 만약 기계에 이상이 있거나 추가적인 지원이 필요하시면 언제든지 연락 주시기 바랍니다."
    },
)
memory.save_context(
    inputs={"human": "감사합니다, 도움이 많이 되었어요!"},
    outputs={
        "ai": "언제든지 도와드릴 준비가 되어 있습니다. 추가적인 질문이나 지원이 필요하시면 언제든지 문의해 주세요. 좋은 하루 되세요!"
    },
)
```

- 최대 토큰의 길이를 **150** 으로 설정하고 대화를 저장했을 때 어떻게 동작하는지 확인해 보겠습니다.

```python
# 대화내용을 확인합니다.
memory.load_memory_variables({})["history"]
```

# 05 ConversationEntityMemory

- 엔티티 메모리는 대화에서 특정 엔티티에 대한 주어진 사실을 기억합니다.

- 엔티티 메모리는 엔티티에 대한 정보를 추출하고(LLM 사용) 시간이 지남에 따라 해당 엔티티에 대한 지식을 축적합니다(역시 LLM 사용).

- 대화를 시작합니다.

- 입력한 대화를 바탕으로 `ConversationEntityMemory` 는 주요 Entity 정보를 별도로 저장합니다.

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# Entity Memory를 사용하는 프롬프트 내용을 출력합니다.
print(ENTITY_MEMORY_CONVERSATION_TEMPLATE.template)

# LLM 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ConversationChain 을 생성합니다.
conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

conversation.predict(
    input="테디와 셜리는 한 회사에서 일하는 동료입니다."
    "테디는 개발자이고 셜리는 디자이너입니다. "
    "그들은 최근 회사에서 일하는 것을 그만두고 자신들의 회사를 차릴 계획을 세우고 있습니다."
)

# entity memory 를 출력합니다.
conversation.memory.entity_store.store
```

# 06.ConversationKGMemory

- 지식 그래프의 힘을 활용하여 정보를 저장하고 불러옵니다.

- 이를 통해 모델이 서로 다른 개체 간의 관계를 이해하는 데 도움을 주고, 복잡한 연결망과 역사적 맥락을 기반으로 대응하는 능력을 향상시킵니다.

```python

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationKGMemory


llm = ChatOpenAI(temperature=0)

memory = ConversationKGMemory(llm=llm, return_messages=True)
memory.save_context(
    {"input": "이쪽은 Pangyo 에 거주중인 김셜리씨 입니다."},
    {"output": "김셜리씨는 누구시죠?"},
)
memory.save_context(
    {"input": "김셜리씨는 우리 회사의 신입 디자이너입니다."},
    {"output": "만나서 반갑습니다."},
)

memory.load_memory_variables({"input": "김셜리씨는 누구입니까?"})
```
---

# 07 ConversationSummaryMemory

- 이제 조금 더 복잡한 메모리 유형인 `ConversationSummaryMemory` 를 사용하는 방법을 살펴 보겠습니다.

- 이 유형의 메모리는 시간 경과에 따른 **대화의 요약** 을 생성합니다. 이는 시간 경과에 따른 대화의 정보를 압축하는 데 유용할 수 있습니다.

- 대화 요약 메모리는 대화가 진행되는 동안 대화를 요약하고 **현재 요약을 메모리에 저장** 합니다.

- 그런 다음 이 메모리를 사용하여 지금까지의 대화 요약을 프롬프트/체인에 삽입할 수 있습니다.

- 이 메모리는 과거 메시지 기록을 프롬프트에 그대로 보관하면 토큰을 너무 많이 차지할 수 있는 긴 대화에 가장 유용합니다.

- 저장된 메모리의 history 를 확인합니다.

- 이전의 모든 대화를 압축적으로 요약한 내용을 확인할 수 있습니다.
```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0), return_messages=True)

memory.save_context(
    inputs={"human": "유럽 여행 패키지의 가격은 얼마인가요?"},
    outputs={
        "ai": "유럽 14박 15일 패키지의 기본 가격은 3,500유로입니다. 이 가격에는 항공료, 호텔 숙박비, 지정된 관광지 입장료가 포함되어 있습니다. 추가 비용은 선택하신 옵션 투어나 개인 경비에 따라 달라집니다."
    },
)
memory.save_context(
    inputs={"human": "여행 중에 방문할 주요 관광지는 어디인가요?"},
    outputs={
        "ai": "이 여행에서는 파리의 에펠탑, 로마의 콜로세움, 베를린의 브란덴부르크 문, 취리히의 라이네폴 등 유럽의 유명한 관광지들을 방문합니다. 각 도시의 대표적인 명소들을 포괄적으로 경험하실 수 있습니다."
    },
)
memory.save_context(
    inputs={"human": "여행자 보험은 포함되어 있나요?"},
    outputs={
        "ai": "네, 모든 여행자에게 기본 여행자 보험을 제공합니다. 이 보험은 의료비 지원, 긴급 상황 발생 시 지원 등을 포함합니다. 추가적인 보험 보장을 원하시면 상향 조정이 가능합니다."
    },
)
memory.save_context(
    inputs={
        "human": "항공편 좌석을 비즈니스 클래스로 업그레이드할 수 있나요? 비용은 어떻게 되나요?"
    },
    outputs={
        "ai": "항공편 좌석을 비즈니스 클래스로 업그레이드하는 것이 가능합니다. 업그레이드 비용은 왕복 기준으로 약 1,200유로 추가됩니다. 비즈니스 클래스에서는 더 넓은 좌석, 우수한 기내식, 그리고 추가 수하물 허용량 등의 혜택을 제공합니다."
    },
)
memory.save_context(
    inputs={"human": "패키지에 포함된 호텔의 등급은 어떻게 되나요?"},
    outputs={
        "ai": "이 패키지에는 4성급 호텔 숙박이 포함되어 있습니다. 각 호텔은 편안함과 편의성을 제공하며, 중심지에 위치해 관광지와의 접근성이 좋습니다. 모든 호텔은 우수한 서비스와 편의 시설을 갖추고 있습니다."
    },
)
memory.save_context(
    inputs={"human": "식사 옵션에 대해 더 자세히 알려주실 수 있나요?"},
    outputs={
        "ai": "이 여행 패키지는 매일 아침 호텔에서 제공되는 조식을 포함하고 있습니다. 점심과 저녁 식사는 포함되어 있지 않아, 여행자가 자유롭게 현지의 다양한 음식을 경험할 수 있는 기회를 제공합니다. 또한, 각 도시별로 추천 식당 리스트를 제공하여 현지의 맛을 최대한 즐길 수 있도록 도와드립니다."
    },
)
memory.save_context(
    inputs={"human": "패키지 예약 시 예약금은 얼마인가요? 취소 정책은 어떻게 되나요?"},
    outputs={
        "ai": "패키지 예약 시 500유로의 예약금이 필요합니다. 취소 정책은 예약일로부터 30일 전까지는 전액 환불이 가능하며, 이후 취소 시에는 예약금이 환불되지 않습니다. 여행 시작일로부터 14일 전 취소 시 50%의 비용이 청구되며, 그 이후는 전액 비용이 청구됩니다."
    },
)


```
- 영어로 가지고있음

```python
# 저장된 메모리 확인
print(memory.load_memory_variables({})["history"])
```

## ConversationSummaryBufferMemory

- 최근 n개까지의 데이터는 원본을 유지함
- 그리고 그 이외는 요약을 진행

- `ConversationSummaryBufferMemory` 는 두 가지 아이디어를 결합한 것입니다.

- 최근 대화내용의 버퍼를 메모리에 유지하되, 이전 대화내용을 완전히 플러시(flush)하지 않고 요약으로 컴파일하여 두 가지를 모두 사용합니다.

- 대화내용을 플러시할 시기를 결정하기 위해 상호작용의 개수가 아닌 **토큰 길이** 를 사용합니다.

- 메모리에 저장된 대화를 확인합니다.

- 아직은 대화내용을 요약하지 않습니다. 기준이 되는 **200** 토큰에 도달하지 않았기 때문입니다.


```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

llm = ChatOpenAI()

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,  # 요약의 기준이 되는 토큰 길이를 설정합니다.
    return_messages=True,
)

memory.save_context(
    inputs={"human": "유럽 여행 패키지의 가격은 얼마인가요?"},
    outputs={
        "ai": "유럽 14박 15일 패키지의 기본 가격은 3,500유로입니다. 이 가격에는 항공료, 호텔 숙박비, 지정된 관광지 입장료가 포함되어 있습니다. 추가 비용은 선택하신 옵션 투어나 개인 경비에 따라 달라집니다."
    },
)

# 메모리에 저장된 대화내용 확인
memory.load_memory_variables({})["history"]
```


# 08. VectorStoreRetrieverMemory

- `VectorStoreRetrieverMemory` 는 벡터 스토어에 메모리를 저장하고 호출될 때마다 가장 '눈에 띄는' 상위 K개의 문서를 쿼리합니다.

- 이는 대화내용의 순서를 **명시적으로 추적하지 않는다는 점** 에서 다른 대부분의 메모리 클래스와 다릅니다.


- 1. 먼저, 벡터 스토어를 초기화 합니다.

- 2. 실제 사용에서는 `k`를 더 높은 값으로 설정하지만, 여기서는 `k=1` 을 사용하여 다음과 같이 표시합니다.

- 3. 다음의 질문을 했을 때 Vector Store 로 부터 1개(k=1 이기 때문)의 가장 관련성 높은 대화를 반환합니다.

    - 질문: "면접자 전공은 무엇인가요?"

- 4. 이번에는 다른 질문을 통해 가장 연관성 높은 1개 대화를 추출합니다.

    - 질문: "면접자가 프로젝트에서 맡은 역할은 무엇인가요?"


```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


# 임베딩 모델을 정의합니다.
embeddings_model = OpenAIEmbeddings()

# Vector Store 를 초기화 합니다.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})


from langchain.memory import VectorStoreRetrieverMemory

# 벡터 조회가 여전히 의미적으로 관련성 있는 정보를 반환한다는 것을 보여주기 위해서입니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 임의의 대화를 저장합니다.
memory.save_context(
    inputs={
        "human": "안녕하세요, 오늘 면접에 참석해주셔서 감사합니다. 자기소개 부탁드립니다."
    },
    outputs={
        "ai": "안녕하세요. 저는 컴퓨터 과학을 전공한 신입 개발자입니다. 대학에서는 주로 자바와 파이썬을 사용했으며, 최근에는 웹 개발 프로젝트에 참여하여 실제 사용자를 위한 서비스를 개발하는 경험을 했습니다."
    },
)
memory.save_context(
    inputs={"human": "프로젝트에서 어떤 역할을 맡았나요?"},
    outputs={
        "ai": "제가 맡은 역할은 백엔드 개발자였습니다. 사용자 데이터 처리와 서버 로직 개발을 담당했으며, RESTful API를 구현하여 프론트엔드와의 통신을 담당했습니다. 또한, 데이터베이스 설계에도 참여했습니다."
    },
)
memory.save_context(
    inputs={
        "human": "팀 프로젝트에서 어려움을 겪었던 경험이 있다면 어떻게 해결했나요?"
    },
    outputs={
        "ai": "프로젝트 초기에 의사소통 문제로 몇 가지 어려움이 있었습니다. 이를 해결하기 위해 저희 팀은 정기적인 미팅을 갖고 각자의 진행 상황을 공유했습니다. 또한, 문제가 발생했을 때는 적극적으로 의견을 나누고, 합리적인 해결책을 찾기 위해 노력했습니다."
    },
)
memory.save_context(
    inputs={"human": "개발자로서 자신의 강점은 무엇이라고 생각하나요?"},
    outputs={
        "ai": "제 강점은 빠른 학습 능력과 문제 해결 능력입니다. 새로운 기술이나 도구를 빠르게 습득할 수 있으며, 복잡한 문제에 직면했을 때 창의적인 해결책을 제시할 수 있습니다. 또한, 팀워크를 중시하며 동료들과 협력하는 것을 중요하게 생각합니다."
    },
)

# 메모리에 질문을 통해 가장 연관성 높은 1개 대화를 추출합니다.
print(memory.load_memory_variables({"human": "면접자 전공은 무엇인가요?"})["history"])

print(
    memory.load_memory_variables(
        {"human": "면접자가 프로젝트에서 맡은 역할은 무엇인가요?"}
    )["history"]
)

```
- 시간관계를 고려한 내용은 아니지만, 정보들이 많이 있을때 특정 선택정보를 답변을 찾을때 유용하다.

# 9. LCEL (대화내용 기억하기): 메모리 추가

- 임의의 체인에 메모리를 추가하는 방법을 보여줍니다. 현재 메모리 클래스를 사용할 수 있지만 수동으로 연결해야 합니다

---
1. 대화내용을 저장할 메모리인 `ConversationBufferMemory` 생성하고 `return_messages` 매개변수를 `True`로 설정하여, 생성된 인스턴스가 메시지를 반환하도록 합니다.

- `memory_key` 설정: 추후 Chain 의 `prompt` 안에 대입될 key 입니다. 변경하여 사용할 수 있습니다.
- messagesPlaceholder : 데이터가 진행될수록 이곳에서 저장된다.
---

2. 저장된 대화기록을 확인합니다. 아직 저장하지 않았으므로, 대화기록은 비어 있습니다.

---
3. 대화내용을 저장할 메모리인 `ConversationBufferMemory` 생성하고 `return_messages` 매개변수를 `True`로 설정하여, 생성된 인스턴스가 메시지를 반환하도록 합니다.

- `memory_key` 설정: 추후 Chain 의 `prompt` 안에 대입될 key 입니다. 변경하여 사용할 수 있습니다.

---

4. 저장된 대화기록을 확인합니다. 아직 저장하지 않았으므로, 대화기록은 비어 있습니다.

---

5. `RunnablePassthrough.assign`을 사용하여 `chat_history` 변수에 `memory.load_memory_variables` 함수의 결과를 할당하고, 이 결과에서 `chat_history` 키에 해당하는 값을 추출합니다.

---
6. `runnable` 에 첫 번째 대화를 시작합니다.

- `input`: 사용자 입력 대화가 전달됩니다.
- `chat_history`: 대화 기록이 전달됩니다.

---
7. `memory.save_context` 함수는 입력 데이터(`inputs`)와 응답 내용(`response.content`)을 메모리에 저장하는 역할을 합니다. 이는 AI 모델의 학습 과정에서 현재 상태를 기록하거나, 사용자의 요청과 시스템의 응답을 추적하는 데 사용될 수 있습니다.

---
8. 이름을 기억하고 있는지 추가 질의합니다.

---
```python
from dotenv import load_dotenv

load_dotenv()

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


# ChatOpenAI 모델을 초기화합니다.
model = ChatOpenAI()

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

memory.load_memory_variables({})  # 메모리 변수를 빈 딕셔너리로 초기화합니다.

runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
)

runnable.invoke({"input": "hi"})

runnable.invoke({"input": "hi"})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

runnable.invoke({"input": "hi!"})

chain = runnable | prompt | model

# 첫번째 대화를 진행한다.
# chain 객체의 invoke 메서드를 사용하여 입력에 대한 응답을 생성합니다.
response = chain.invoke({"input": "만나서 반갑습니다. 제 이름은 테디입니다."})
print(response.content)  # 생성된 응답을 출력합니다.


memory.load_memory_variables({})

# 입력된 데이터와 응답 내용을 메모리에 저장합니다.
memory.save_context(
    {"human": "만나서 반갑습니다. 제 이름은 테디입니다."}, {"ai": response.content}
)

# 저장된 대화기록을 출력합니다.
memory.load_memory_variables({})

# 이름을 기억하고 있는지 추가 질의합니다.
response = chain.invoke({"input": "제 이름이 무엇이었는지 기억하세요?"})
# 답변을 출력합니다.
print(response.content)

# 이름을 기억하고 있는지 추가 질의합니다.
response = chain.invoke({"input": "제 이름이 무엇이었는지 기억하세요?"})
# 답변을 출력합니다.
print(response.content)


```
### 이거 논문 쌉가능

## 커스텀 ConversationChain 구현 예시

```python

from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ChatOpenAI 모델을 초기화합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

class MyConversationChain(Runnable):

    def __init__(self, llm, prompt, memory, input_key="input"):

        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)  # memory_key 와 동일하게 입력합니다.
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer


# ChatOpenAI 모델을 초기화합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# 요약 메모리로 교체할 경우
# memory = ConversationSummaryMemory(
#     llm=llm, return_messages=True, memory_key="chat_history"
# )

conversation_chain = MyConversationChain(llm, prompt, memory)

conversation_chain.invoke("안녕하세요? 만나서 반갑습니다. 제 이름은 테디 입니다.")

conversation_chain.invoke("제 이름이 뭐라고요?")

conversation_chain.invoke("앞으로는 영어로만 답변해주세요 알겠어요?")

conversation_chain.invoke("제 이름을 다시 한 번 말해주세요")

conversation_chain.memory.load_memory_variables({})["chat_history"]
```

# 10. SQL (SQLAlchemy)

> [Structured Query Language (SQL)](https://en.wikipedia.org/wiki/SQL)은 프로그래밍에 사용되는 도메인 특화 언어로, 관계형 데이터베이스 관리 시스템(RDBMS)에서 데이터를 관리하거나 관계형 데이터 스트림 관리 시스템(RDSMS)에서 스트림 처리를 위해 설계되었습니다. 특히 엔티티와 변수 간의 관계를 포함하는 구조화된 데이터를 다루는 데 유용합니다.

> [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)는 MIT 라이선스에 따라 배포되는 Python 프로그래밍 언어용 오픈 소스 `SQL` 툴킷이자 객체 관계 매퍼(ORM)입니다.

이 노트북에서는 `SQLAlchemy`가 지원하는 모든 데이터베이스에 채팅 기록을 저장할 수 있는 `SQLChatMessageHistory` 클래스에 대해 설명합니다.

`SQLite` 이외의 데이터베이스와 함께 사용하려면 해당 데이터베이스 드라이버를 설치해야 합니다.
ß