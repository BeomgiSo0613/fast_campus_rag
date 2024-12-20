# 출력파서(outputParsar)

## 출력파서(output Parser)

- 출력파서를 일정하게 배출해 주어야 추후 chain을 연결하는데 있어 큰 도움이 된더.
- 다양한 outputparser에 대해 알아보자


## PydanticOutputOarsar

`PydanticOutputParser` (대부분의 OutputParser에 해당)에는 주로 **두 가지 핵심 메서드**가 구현되어야 합니다.

- **`get_format_instructions()`**: 언어 모델이 출력해야 할 정보의 형식을 정의하는 지침을 제공합니다. 예를 들어, 언어 모델이 출력해야 할 데이터의 필드와 그 형태를 설명하는 지침을 문자열로 반환할 수 있습니다. 이 지침은 언어 모델이 출력을 구조화하고 특정 데이터 모델에 맞게 변환하는 데 매우 중요합니다.
    - 지침에 맞춰서 답변을 해줘!!

- **`parse()`**: 언어 모델의 출력(문자열로 가정)을 받아 이를 특정 구조로 분석하고 변환합니다. Pydantic과 같은 도구를 사용하여 입력된 문자열을 사전 정의된 스키마에 따라 검증하고, 해당 스키마를 따르는 데이터 구조로 변환합니다.

- from pydantic import BaseModel, Field


```python
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=EmailSummary)

parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Please answer the following questions in KOREAN.

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)

# format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
prompt = prompt.partial(format=parser.get_format_instructions())

# chain 을 생성합니다.
chain = prompt | llm

# chain 을 실행하고 결과를 출력합니다.
response = chain.stream(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)

# 결과는 JSON 형태로 출력됩니다.
output = stream_response(response, return_output=True)

# PydanticOutputParser 를 사용하여 결과를 파싱합니다.
structured_output = parser.parse(output)
print(structured_output)
```

-  parser 가 추가된 체인 생성

    - 출력 결과를 정의한 Pydantic 객체로 생성할 수 있습니다.
```python
# 출력 파서를 추가하여 전체 체인을 재구성합니다.
chain = prompt | llm | parser

# chain 을 실행하고 결과를 출력합니다.
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)

# 결과는 EmailSummary 객체 형태로 출력됩니다.
response
```

- with_structured_output()
    - `.with_structured_output(Pydantic)`을 사용하여 출력 파서를 추가하면, 출력을 Pydantic 객체로 변환할 수 있습니다.

- 프롬프트를 정의합니다.
    1. `question`: 유저의 질문을 받습니다.
    2. `email_conversation`: 이메일 본문의 내용을 입력합니다.
    3. `format`: 형식을 지정합니다.

```python
llm_with_structered = ChatOpenAI(
    temperature=0, model_name="gpt-4o"
).with_structured_output(EmailSummary)

# invoke() 함수를 호출하여 결과를 출력합니다.
answer = llm_with_structered.invoke(email_conversation)
answer
# 지정한 객체 형식으로 답을 확인할 수 있다.
answer.email
answer.summary
```
### LangSmith에서 OutputParsar

---

## CommaSeparatedListOutputParser

- `CommaSeparatedListOutputParser`는 쉼표로 구분된 항목 목록을 반환할 필요가 있을 때 유용한 출력 파서입니다.

- 이 파서를 사용하면, 입력된 데이터나 요청된 정보를 쉼표로 구분하여 명확하고 간결한 목록 형태로 제공할 수 있습니다. 예를 들어, 여러 개의 데이터 포인트, 이름, 항목 또는 다양한 값을 나열할 때 효과적으로 정보를 정리하고 사용자에게 전달할 수 있습니다.

- 이 방법은 정보를 구조화하고 가독성을 높이며, 특히 데이터를 다루거나 리스트 형태의 결과를 요구하는 경우에 매우 유용합니다.





### cf) 파이썬

- 파이썬에서 클래스는 **객체지향 프로그래밍(OOP)**의 기본 구성 요소로, 데이터(속성)와 행동(메서드)을 하나의 단위로 묶는 역할을 합니다. 클래스를 사용하면 코드의 재사용성, 유지보수성, 캡슐화를 높일 수 있습니다.

```python
class User:
    name: str  # name은 문자열 타입
    age: int   # age는 정수 타입

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        return f"안녕하세요, {self.name}입니다. 나이는 {self.age}살입니다."
```

- name: str와 age: int는 속성의 데이터 타입을 나타냅니다.
- 메서드의 반환 타입도 -> str로 지정할 수 있습니다.

### 파이썬 클래스에서 상속
- **상속(Inheritance)**은 객체지향 프로그래밍(OOP)의 핵심 개념 중 하나로, 기존 클래스(부모 클래스 또는 기반 클래스 )의 속성과 메서드를 새로운 클래스(자식 클래스 또는 파생 클래스)가 물려받아 사용할 수 있게 해줍니다.
상속을 사용하면 코드 재사용성을 높이고, 기존 코드를 수정하지 않고도 확장할 수 있습니다.



```python
# 부모 클래스
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self):
        print(f"{self.name}은(는) 소리를 냅니다.")

# 자식 클래스
class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)  # 부모 클래스의 초기화 메서드 호출
        self.breed = breed  # 자식 클래스만의 속성 추가

    def speak(self):  # 부모 메서드 재정의 (오버라이딩)
        print(f"{self.name}은(는) 멍멍 소리를 냅니다.")

    def fetch(self):
        print(f"{self.name}이(가) 물건을 가져옵니다.")

# 부모 클래스 사용
animal = Animal("동물")
animal.speak()  # 동물은(는) 소리를 냅니다.

# 자식 클래스 사용
dog = Dog("바둑이", "진돗개")
dog.speak()  # 바둑이는(는) 멍멍 소리를 냅니다.
dog.fetch()  # 바둑이가 물건을 가져옵니다
```

```plaintext
동물은(는) 소리를 냅니다.
바둑이는(는) 멍멍 소리를 냅니다.
바둑이가 물건을 가져옵니다.
```

1. super() 키워드
- 자식 클래스에서 부모 클래스의 메서드(특히 __init__)를 호출할 때 사용합니다.

```python
super().__init__(name)
```

2. 메서드 오버라이딩(Method Overriding)
- 자식 클래스에서 부모 클래스의 메서드를 재정의할 수 있습니다.
    - 예: Dog 클래스의 speak 메서드가 Animal 클래스의 speak를 덮어씌움.

3. 추가 기능 확장
- 자식 클래스는 부모 클래스에 없는 새로운 속성과 메서드를 추가할 수 있습니다.

## 05 콤마로 구분된 리스트 출력 파서

# CommaSeparatedListOutputParser

`CommaSeparatedListOutputParser`는 쉼표로 구분된 항목 목록을 반환할 필요가 있을 때 유용한 출력 파서입니다.

이 파서를 사용하면, 입력된 데이터나 요청된 정보를 쉼표로 구분하여 명확하고 간결한 목록 형태로 제공할 수 있습니다. 예를 들어, 여러 개의 데이터 포인트, 이름, 항목 또는 다양한 값을 나열할 때 효과적으로 정보를 정리하고 사용자에게 전달할 수 있습니다.

이 방법은 정보를 구조화하고 가독성을 높이며, 특히 데이터를 다루거나 리스트 형태의 결과를 요구하는 경우에 매우 유용합니다.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 콤마로 구분된 리스트 출력 파서 초기화
output_parser = CommaSeparatedListOutputParser()

# 출력 형식 지침 가져오기
format_instructions = output_parser.get_format_instructions()
# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    # 주제에 대한 다섯 가지를 나열하라는 템플릿
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    # 부분 변수로 형식 지침 사용
    partial_variables={"format_instructions": format_instructions},
)

# ChatOpenAI 모델 초기화
model = ChatOpenAI(temperature=0)

# 프롬프트, 모델, 출력 파서를 연결하여 체인 생성
chain = prompt | model | output_parser

# "대한민국 관광명소"에 대한 체인 호출 실행
chain.invoke({"subject": "대한민국 관광명소"})

# 스트림을 순회합니다.
for s in chain.stream({"subject": "대한민국 관광명소"}):
    print(s)  # 스트림의 내용을 출력합니다.
```

### 06.구조화된 출력파서

- StructuredOutputParser

- StructuredOutputParser는 LLM에 대한 답변을 `dict` 형식으로 구성하고, key/value 쌍으로 여러 필드를 반환하고자 할 때 유용하게 사용할 수 있습니다. 

    - 장점

- Pydantic/JSON 파서가 더 강력하다는 평가를 받지만, StructuredOutputParser는 로컬 모델과 같은 덜 강력한 모델에서도 유용합니다. 이는 GPT나 Claude 모델보다 인텔리전스가 낮은(즉, parameter 수가 적은) 모델에서 특히 효과적입니다. 
    - 강력하다 : 우리가 원하는 구조화된 결과를 받을 수 있다.

    - 참고 사항

- 로컬 모델의 경우 `Pydantic` 파서가 동작하지 않는 상황이 빈번하게 발생할 수 있습니다. 이러한 경우, 대안으로 StructuredOutputParser를 사용하는 것이 좋은 해결책이 될 수 있습니다.

```python

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

```

- `ResponseSchema` 클래스를 사용하여 사용자의 질문에 대한 답변과 사용된 소스(웹사이트)에 대한 설명을 포함하는 응답 스키마를 정의합니다.
- `StructuredOutputParser`를 `response_schemas`를 사용하여 초기화하여, 정의된 응답 스키마에 따라 출력을 구조화합니다.

```python

# 사용자의 질문에 대한 답변
response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 `출처`, `웹사이트주소` 이여야 합니다.",
    ),
]
# 응답 스키마를 기반으로 한 구조화된 출력 파서 초기화
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

```
- 이제 응답이 어떻게 포맷되어야 하는지에 대한 지시사항이 포함된 문자열을 받게 되며(schemas), 정의된 스키마를 프롬프트에 삽입합니다.

```python

# 출력 형식 지시사항을 파싱합니다.
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)  # ChatOpenAI 모델 초기화
chain = prompt | model | output_parser  # 프롬프트, 모델, 출력 파서를 연결

# 대한민국의 수도가 무엇인지 질문합니다.
answer = chain.invoke({"question": "대한민국의 수도는 어디인가요?"})

type(answer)

answer["source"]

for s in chain.stream({"question": "세종대왕의 업적은 무엇인가요?"}):
    # 스트리밍 출력
    print(s)
```

### 7.JSON형식 출력파서

#### JsonOutputParser

JsonOutputParser는 사용자가 원하는 JSON 스키마를 지정할 수 있게 해주는 도구입니다. 이 도구는 Large Language Model (LLM)이 데이터를 조회하고 결과를 도출할 때, 지정된 스키마에 맞게 JSON 형식으로 데이터를 반환할 수 있도록 설계되었습니다.

LLM이 데이터를 정확하고 효율적으로 처리하여 사용자가 원하는 형태의 JSON을 생성하기 위해서는, 모델의 용량(예: 인텔리전스)이 충분히 커야 합니다. 예를 들어, llama-70B 모델은 llama-8B 모델보다 더 큰 용량을 가지고 있어 보다 복잡한 데이터를 처리하는 데 유리합니다.

**[참고]**

`JSON (JavaScript Object Notation)` 은 데이터를 저장하고 구조적으로 전달하기 위해 사용되는 경량의 데이터 교환 포맷입니다. 웹 개발에서 매우 중요한 역할을 하며, 서버와 클라이언트 간의 통신을 위해 널리 사용됩니다. JSON은 읽기 쉽고, 기계가 파싱하고 생성하기 쉬운 텍스트를 기반으로 합니다.

JSON의 기본 구조
JSON 데이터는 이름(키)과 값의 쌍으로 이루어져 있습니다. 여기서 "이름"은 문자열이고, "값"은 다양한 데이터 유형일 수 있습니다. JSON은 두 가지 기본 구조를 가집니다:

- 객체: 중괄호 {}로 둘러싸인 키-값 쌍의 집합입니다. 각 키는 콜론 :을 사용하여 해당하는 값과 연결되며, 여러 키-값 쌍은 쉼표 ,로 구분됩니다.
- 배열: 대괄호 []로 둘러싸인 값의 순서 있는 목록입니다. 배열 내의 값은 쉼표 ,로 구분됩니다.

```json
{
  "name": "John Doe",
  "age": 30,
  "is_student": false,
  "skills": ["Java", "Python", "JavaScript"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown"
  }
}
```

- `JsonOutputParser`를 사용하여 파서를 설정하고, 프롬프트 템플릿에 지시사항을 주입합니다.


```python

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# OpenAI 객체를 생성합니다.
model = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 원하는 데이터 구조를 정의합니다.
class Topic(BaseModel):
    description: str = Field(description="주제에 대한 간결한 설명")
    hashtags: str = Field(description="해시태그 형식의 키워드(2개 이상)")

# 질의 작성
question = "지구 온난화의 심각성 대해 알려주세요."

# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
parser = JsonOutputParser(pydantic_object=Topic)
print(parser.get_format_instructions())

# 프롬프트 템플릿을 설정합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트 입니다. 질문에 간결하게 답변하세요."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# 체인을 구성합니다.
chain = prompt | model | parser

# 체인을 호출하여 쿼리 실행
answer = chain.invoke({"question": question})

# 타입을 확인합니다.
type(answer)

# answer 객체를 출력합니다.
answer
```

### 8. Pandas DataFrame

#### PandasDataFrameOutputParser

- **Pandas DataFrame**은 Python 프로그래밍 언어에서 널리 사용되는 데이터 구조로, 데이터 조작 및 분석을 위한 강력한 도구입니다. DataFrame은 구조화된 데이터를 효과적으로 다루기 위한 포괄적인 도구 세트를 제공하며, 이를 통해 데이터 정제, 변환 및 분석과 같은 다양한 작업을 수행할 수 있습니다.

- 이 **출력 파서**는 사용자가 임의의 Pandas DataFrame을 지정하여 해당 DataFrame에서 데이터를 추출하고, 이를 형식화된 사전(dictionary) 형태로 조회할 수 있게 해주는 LLM(Large Language Model) 기반 도구입니다.


- `format_parser_output` 함수는 파서 출력을 사전 형식으로 변환하고 출력 형식을 지정하는 데 사용됩니다. 

```
The output should be formatted as a string as the operation, followed by a colon, followed by the column or row to be queried on, followed by optional array parameters.
1. The column names are limited to the possible columns below.
2. Arrays must either be a comma-separated list of numbers formatted as [1,3,5], or it must be in range of numbers formatted as [0..4].
3. Remember that arrays are optional and not necessarily required.
4. If the column is not in the possible columns or the operation is not a valid Pandas DataFrame operation, return why it is invalid as a sentence starting with either "Invalid column" or "Invalid operation".

As an example, for the formats:
1. String "column:num_legs" is a well-formatted instance which gets the column num_legs, where num_legs is a possible column.
2. String "row:1" is a well-formatted instance which gets row 1.
3. String "column:num_legs[1,2]" is a well-formatted instance which gets the column num_legs for rows 1 and 2, where num_legs is a possible column.
4. String "row:1[num_legs]" is a well-formatted instance which gets row 1, but for just column num_legs, where num_legs is a possible column.
5. String "mean:num_legs[1..3]" is a well-formatted instance which takes the mean of num_legs from rows 1 to 3, where num_legs is a possible column and mean is a valid Pandas DataFrame operation.
6. String "do_something:num_legs" is a badly-formatted instance, where do_something is not a valid Pandas DataFrame operation.
7. String "mean:invalid_col" is a badly-formatted instance, where invalid_col is not a possible column.

Here are the possible columns:
```
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
```

```

- 컬럼에 대한 값을 조회하는 예제입니다.

### 7. DatetimeOutputParser

- `DatetimeOutputParser` 는 LLM의 출력을 `datetime` 형식으로 파싱하는 데 사용할 수 있습니다.


**참고**

| 형식 코드 | 설명                | 예시          |
|------------|---------------------|---------------|
| %Y         | 4자리 연도          | 2024          |
| %y         | 2자리 연도          | 24            |
| %m         | 2자리 월            | 07            |
| %d         | 2자리 일            | 04            |
| %H         | 24시간제 시간       | 14            |
| %I         | 12시간제 시간       | 02            |
| %p         | AM 또는 PM          | PM            |
| %M         | 2자리 분            | 45            |
| %S         | 2자리 초            | 08            |
| %f         | 마이크로초 (6자리)  | 000123        |
| %z         | UTC 오프셋          | +0900         |
| %Z         | 시간대 이름         | KST           |
| %a         | 요일 약어           | Thu           |
| %A         | 요일 전체           | Thursday      |
| %b         | 월 약어             | Jul           |
| %B         | 월 전체             | July          |
| %c         | 전체 날짜와 시간     | Thu Jul  4 14:45:08 2024 |
| %x         | 전체 날짜           | 07/04/24      |
| %X         | 전체 시간           | 14:45:08      |


```python
from dotenv import load_dotenv

load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-OutputParser")


from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 날짜 및 시간 출력 파서
output_parser = DatetimeOutputParser()
output_parser.format = "%Y-%m-%d"

# 사용자 질문에 대한 답변 템플릿
template = """Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    },  # 지침을 템플릿에 적용
)

# 프롬프트 내용을 출력
prompt

# Chain 을 생성합니다.
chain = prompt | ChatOpenAI() | output_parser

# 체인을 호출하여 질문에 대한 답변을 받습니다.
output = chain.invoke({"question": "Google 이 창업한 연도"})

# 결과를 문자열로 변환
output.strftime("%Y-%m-%d")
```

---
# 10. EnumOutputParser

LangChain의 EnumOutputParser는 언어 모델의 출력을 미리 정의된 열거형(Enum) 값 중 하나로 파싱하는 도구입니다. 이 파서의 주요 특징과 사용법은 다음과 같습니다.

## 주요 특징

- **열거형 파싱**: 문자열 출력을 미리 정의된 Enum 값으로 변환합니다.
- **타입 안전성**: 파싱된 결과가 반드시 정의된 Enum 값 중 하나임을 보장합니다.
- **유연성**: 공백이나 줄바꿈 문자를 자동으로 처리합니다.

## 사용 방법

EnumOutputParser는 언어 모델의 출력에서 유효한 Enum 값을 추출하는 데 유용합니다. 이를 통해 출력 데이터의 일관성을 유지하고 예측 가능성을 높일 수 있습니다. 파서를 사용하려면, 미리 정의된 Enum 값을 설정하고 해당 값을 기준으로 문자열 출력을 파싱합니다.

- `enum` 모듈을 사용하여 `Colors` 클래스를 정의합니다.
- `Colors` 클래스는 `Enum`을 상속받으며, `RED`, `GREEN`, `BLUE` 세 가지 색상 값을 가집니다.
- 데이터들의 관계가 동등한 관계여야한다. 범주가 같아야한다.