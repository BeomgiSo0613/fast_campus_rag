# 모델(model)

## 01. RAG에서의 LLM(Large Language Model)

- LLM.pdf
- Chatgpt뿐만아니라 다양한 모델들을 쓸 수 있다.
- Solar, GPT, HuggingFace 등등 다양한 모델
- 과금때문에 모델을 변경한다.

- gpt -> api식
- 직접 다운받로드해서 진행
- open source모델에 서버를 통해 사용
- gpt경우에는 보안 문제가 생길 수도 있다.

- youtube에서도 무료 모델로 많이 사용할 수 있다.
- Opensource모델을 사용한다고 해도 코드변화는 크게 없다.

## 02. 다양한 LLM과 활용모델

- API키 발금
    - gpt이미 받음

- gpt

---
### OpenAI

### 개요
OpenAI는 채팅 전용 Large Language Model (LLM)을 제공합니다. 이 모델을 생성할 때 다양한 옵션을 지정할 수 있으며, 이러한 옵션들은 모델의 동작 방식에 영향을 미칩니다.

### 옵션 상세 설명

`temperature`

- 샘플링 온도를 설정하는 옵션입니다. 값은 0과 2 사이에서 선택할 수 있습니다. 높은 값(예: 0.8)은 출력을 더 무작위하게 만들고, 낮은 값(예: 0.2)은 출력을 더 집중되고 결정론적으로 만듭니다.

`max_tokens`

- 채팅 완성에서 생성할 토큰의 최대 개수를 지정합니다. 이 옵션은 모델이 한 번에 생성할 수 있는 텍스트의 길이를 제어합니다.

`model_name`

- 적용 가능한 모델을 선택하는 옵션입니다. 더 자세한 정보는 [OpenAI 모델 문서](https://platform.openai.com/docs/models)에서 확인할 수 있습니다.


**모델 스펙**

- 링크: https://platform.openai.com/docs/models/gpt-4o

| 모델명 | 설명 | 컨텍스트 길이 | 최대 출력 토큰 | 학습 데이터 |
|--------|------|---------------|-----------------|-------------|
| gpt-4o | GPT-4 터보보다 저렴하고 빠른 최신 다중모드 플래그십 모델 | 128,000 토큰 | 4,096 토큰 | 2023년 10월까지 |
| gpt-4-turbo | 최신 GPT-4 터보 모델. 비전 기능, JSON 모드, 기능 호출 지원 | 128,000 토큰 | 4,096 토큰 | 2023년 12월까지 |
| gpt-4o-mini | GPT-3.5 터보보다 더 우수한 성능의 작은 모델 | 128,000 토큰 | 16,384 토큰 | 2023년 10월까지 |
| o1-preview | 다양한 도메인의 어려운 문제 해결을 위한 추론 모델 | 128,000 토큰 | 32,768 토큰 | 2023년 10월까지 |
| o1-mini | 코딩, 수학, 과학에 특화된 빠른 추론 모델 | 128,000 토큰 | 65,536 토큰 | 2023년 10월까지 |
| gpt-4o-realtime | 실시간 API용 오디오 및 텍스트 입력 처리 모델 (베타) | 128,000 토큰 | 4,096 토큰 | 2023년 10월까지 |
---

### Anthropic_Cloud

### Cohere


### Upstage

### Xionic

- 무료로 사용가능하다.

### LogicKor을 통해 밴치마킹으로 확인 가능함
- https://lk.instruct.kr/


## 캐싱(Caching)

- LangChain은 LLM을 위한 선택적 캐싱 레이어를 제공합니다.

- 이는 두 가지 이유로 유용합니다.

    - 동일한 완료를 여러 번 요청하는 경우 LLM 공급자에 대한 **API 호출 횟수를 줄여 비용을 절감**할 수 있습니다.
    - LLM 제공업체에 대한 **API 호출 횟수를 줄여 애플리케이션의 속도를 높일 수** 있습니다.

### InMemoryCache

- 인메모리 캐시를 사용하여 동일 질문에 대한 답변을 저장하고, 캐시에 저장된 답변을 반환합니다.
```python

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH04-Models")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 모델을 생성합니다.
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template("{country} 에 대해서 200자 내외로 요약해줘")

# 체인을 생성합니다.
chain = prompt | llm

%%time 
response = chain.invoke({"country": "한국"})
print(response.content)
```

- 고객센터나 일반적인 답변을 계속이야기하는경우는 케시를 사용하는게 좋아!

### SQLite Cache

- 휘발되지 않은 메모리


```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import os

# 캐시 디렉토리를 생성합니다.
if not os.path.exists("cache"):
    os.makedirs("cache")

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))

%%time 
# 체인을 실행합니다.
response = chain.invoke({"country": "한국"})
print(response.content)

```

## 04. 직렬화(Serialization)

### 직렬화(Serialization) 란?

- Goal : 우리가 만들어놓은 prompt들을 

- 모델을 저장함이 주된 목적이다.
    - chain -> invoke -> 결과
    - chain의 저장 확장자가 애매하다.
    - chain이라는 것을 직렬화하는 과정을 통해서 -> JSON형식으로 만들어 바꿔 사용한다.
- 그 반대 개념을 역 직렬화라고한다.

- 항상 가능한건 아니다!

1. **정의:**
   - 모델을 저장 가능한 형식으로 변환하는 과정

2. **목적:**
   - 모델 재사용 (재훈련 없이)
   - 모델 배포 및 공유 용이
   - 계산 리소스 절약

3. **장점:**
   - 빠른 모델 로딩
   - 버전 관리 가능
   - 다양한 환경에서 사용 가능

모델 직렬화는 AI 개발 및 배포 과정에서 중요한 단계로, 효율적인 모델 관리와 재사용을 가능하게 합니다.

`is_lc_serializable` 클래스 메서드로 실행하여 LangChain 클래스가 직렬화 가능한지 확인할 수 있습니다.


```python
# 직렬화가 가능한지 체크합니다.
print(f"ChatOpenAI: {ChatOpenAI.is_lc_serializable()}")

# 체인을 생성합니다.
chain = prompt | llm

# 직렬화가 가능한지 체크합니다.
chain.is_lc_serializable()
```

### 체인(Chain) 직렬화(dumps, dumpd)

#### 개요

- 체인 직렬화는 직렬화 가능한 모든 객체를 딕셔너리 또는 JSON 문자열로 변환하는 과정을 의미합니다.

#### 직렬화 방법

- 객체의 속성 및 데이터를 키-값 쌍으로 저장하여 딕셔너리 형태로 변환합니다.

- 이러한 직렬화 방식은 객체를 쉽게 저장하고 전송할 수 있게 하며, 다양한 환경에서 객체를 재구성할 수 있도록 합니다.

**참고**
- `dumps`: 객체를 JSON 문자열로 직렬화
- `dumpd`: 객체를 딕셔너리로 직렬화

```python
from langchain_core.load import dumpd, dumps

dumpd_chain = dumpd(chain)
dumpd_chain

# 직렬화된 체인의 타입을 확인합니다.
type(dumpd_chain)

# dumps 함수를 사용하여 직렬화된 체인을 확인합니다.
dumps_chain = dumps(chain)
dumps_chain

# 직렬화된 체인의 타입을 확인합니다.
type(dumps_chain)
```

### Pickle 파일

    ### 개요

    Pickle 파일은 Python 객체를 바이너리 형태로 직렬화하는 포맷입니다.

    ### 특징

    1. **형식:**
    - Python 객체를 바이너리 형태로 직렬화하는 포맷

    2. **특징:**
    - Python 전용 (다른 언어와 호환 불가)
    - 대부분의 Python 데이터 타입 지원 (리스트, 딕셔너리, 클래스 등)
    - 객체의 상태와 구조를 그대로 보존

    3. **장점:**
    - 효율적인 저장 및 전송
    - 복잡한 객체 구조 유지
    - 빠른 직렬화/역직렬화 속도

    4. **단점:**
    - 보안 위험 (신뢰할 수 없는 데이터 역직렬화 시 주의 필요)
    - 사람이 읽을 수 없는 바이너리 형식

    ### 주요 용도

    1. 객체 캐싱
    2. 머신러닝 모델 저장
    3. 프로그램 상태 저장 및 복원

    ### 사용법

    - `pickle.dump()`: 객체를 파일에 저장
    - `pickle.load()`: 파일에서 객체 로드

```python
import pickle

# fuit_chain.pkl 파일로 직렬화된 체인을 저장합니다.
with open("fruit_chain.pkl", "wb") as f:
    pickle.dump(dumpd_chain, f)

import json

with open("fruit_chain.json", "w") as fp:
    json.dump(dumpd_chain, fp)

import pickle

# pickle 파일을 로드합니다.
with open("fruit_chain.pkl", "rb") as f:
    loaded_chain = pickle.load(f)


from langchain_core.load import load

# 체인을 로드합니다.
chain_from_file = load(loaded_chain)

# 체인을 실행합니다.
print(chain_from_file.invoke({"fruit": "사과"}))


from langchain_core.load import load, loads

# secrets_map API키 불러오기
load_chain = load(
    loaded_chain, secrets_map={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
)

# 불러온 체인이 정상 동작하는지 확인합니다.
load_chain.invoke({"fruit": "사과"})

with open("fruit_chain.json", "r") as fp:
    loaded_from_json_chain = json.load(fp)
    loads_chain = load(loaded_from_json_chain)


# 불러온 체인이 정상 동작하는지 확인합니다.
loads_chain.invoke({"fruit": "사과"})
```

### 토큰 사용량 확인

- 특정 호출에 대한 토큰 사용량을 추적하는 방법에 대해 설명합니다.

- 이 기능은 현재 OpenAI API 에만 구현되어 있습니다.

- 먼저 단일 Chat 모델 호출에 대한 토큰 사용량을 추적하는 매우 간단한 예를 살펴보겠습니다.

```python
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# 모델을 불러옵니다.
llm = ChatOpenAI(model_name="gpt-4o")

from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# 모델을 불러옵니다.
llm = ChatOpenAI(model_name="gpt-4o")

from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# 모델을 불러옵니다.
llm = ChatOpenAI(model_name="gpt-4o")
```

### Google AI chat models (gemini-pro)

Google AI의 `gemini`와 `gemini-vision` 모델뿐만 아니라 다른 생성 모델에 접근하려면 [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) 통합 패키지의 `ChatGoogleGenerativeAI` 클래스를 사용하면 됩니다.

#### API KEY 발급받기

- [링크](https://makersuite.google.com/app/apikey?hl=ko) 에서 API KEY를 발급받아주세요.
- 사용자의 Google API 키를 환경 변수 `GOOGLE_API_KEY`로 설정합니다.

#### langchain_google_genai 패키지에서 ChatGoogleGenerativeAI 클래스를 가져옵니다.

- ChatGoogleGenerativeAI 클래스는 Google의 Generative AI 모델을 사용하여 대화형 AI 시스템을 구현하는 데 사용됩니다.
- 이 클래스를 통해 사용자는 Google의 대화형 AI 모델과 상호 작용할 수 있습니다.
- 모델과의 대화는 채팅 형식으로 이루어지며, 사용자의 입력에 따라 모델이 적절한 응답을 생성합니다.
- ChatGoogleGenerativeAI 클래스는 LangChain 프레임워크와 통합되어 있어, 다른 LangChain 컴포넌트와 함께 사용할 수 있습니다.

    - 지원되는 모델 정보: https://ai.google.dev/gemini-api/docs/models/gemini?hl=ko

## Safety Settings

- Gemini 모델에는 기본 안전 설정(Satety Settings) 이 있지만, 이를 재정의할 수 있습니다.

- 만약 모델로부터 많은 "Safety Warnings"를 받고 있다면, 모델의 `safety_settings` 속성을 조정해 볼 수 있습니다.

- Google의 [Safety Setting Types](https://ai.google.dev/api/python/google/generativeai/types/SafetySettingDict) 문서에서는 사용 가능한 카테고리와 임계값에 대한 열거형 정보를 제공합니다.

- 이 문서에는 콘텐츠 필터링 및 안전 설정과 관련된 다양한 카테고리와 해당 임계값이 정의되어 있어, 개발자들이 생성형 AI 모델을 활용할 때 적절한 안전 설정을 선택하고 적용하는 데 도움을 줍니다.

- 이를 통해 개발자들은 모델이 생성하는 콘텐츠의 안전성과 적절성을 보장하고, 사용자에게 유해하거나 부적절한 내용이 노출되는 것을 방지할 수 있습니다.

```python
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

llm = ChatGoogleGenerativeAI(
    # 사용할 모델을 "gemini-pro"로 지정합니다.
    model="gemini-1.5-pro-latest",
    safety_settings={
        # 위험한 콘텐츠에 대한 차단 임계값을 설정합니다.
        # 이 경우 위험한 콘텐츠를 차단하지 않도록 설정되어 있습니다. (그럼에도 기본적인 차단이 있을 수 있습니다.)
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    # 사용할 모델을 "gemini-pro"로 지정합니다.
    model="gemini-1.5-pro-latest",
)

results = llm.batch(
    [
        "대한민국의 수도는?",
        "대한민국의 주요 관광지 5곳을 나열하세요",
    ]
)

for res in results:
    # 각 결과의 내용을 출력합니다.
    print(res.content)
```

## Multimodal 모델

- `langchain-teddynote` 에서 구현한 멀티모달 모델에 `gemini-1.5-pro` 모델을 활용하여 이미지를 텍스트로 변환 가능합니다.


```python
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

# 객체 생성
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

system_prompt = (
    "당신은 시인입니다. 당신의 임무는 주어진 이미지를 가지고 시를 작성하는 것입니다."
)

user_prompt = "다음의 이미지에 대한 시를 작성해주세요."

# 멀티모달 객체 생성
multimodal_gemini = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)
```


## 11. Ollama 설치 및 Modelifle 설정

- GGUF PDF파일
- GGUF 언어모델을 사용하기 편하기 쉽게 
- 올라마에서 구동하기 위해서는 GGUF 확장자가 필요하다.

- modelfiles
    - PARAMETER stop -> <s> 토큰의 시작, </s> 끝임을 알려주는것
    - FROM 모델명 변경

- 모델 설치 리스트
    - ollama list

- 모델설치
    - ollama pull 모델이름(gemma:7b) / llava:7b 

- 모델실행
    - ollama run (모델명)

- 모델 종료
    /bye

- 모델 직접 설치된 내용 사용하기
    - ollama create EEVE-Korean-s.8b -f Modelfile
    - Modelfile 모델 탬플릿을 잘 설정해주어야 성능이 좋아침

- Langchin

```python
# Ollama 모델을 불러옵니다.
# ollama list에 있는 이름사용
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
```



# GPT4All

![](../04-Model//images/gpt4all.png)

[GitHub:nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) 은 코드, 채팅 형식의 대화를 포함한 방대한 양의 데이터로 학습된 오픈 소스 챗봇 생태계입니다.

이 예제에서는 LangChain을 사용하여 `GPT4All` 모델과 상호 작용하는 방법에 대해 설명합니다.

## 설치방법

1. 먼저, 공식 홈페이지에 접속하여 설치파일을 다운로드 받아 설치합니다
2. [공식 홈페이지](https://gpt4all.io/index.html) 바로가기
3. 파이썬 패키지를 설치합니다.
4. [pip 를 활용한 설치 방법](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-bindings/python/README.md)

- 속도적인 측면에서는 allama가 더 빠르다.
