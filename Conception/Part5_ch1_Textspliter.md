# Part5 Retrievel_Augmented_Generation
# Chapter 1 Text Splitter

## 01. 텍스트 분할의 개념과 중요성. 다양한 전략의 활용
- pdf : Text Spliter
    1. GPT, LLM 등 입력을 받을 수 있는 token의 제한이 있기떄문에, 필요한 정보를 압축해서 넣어주어야한다.
    2. GPT를 압축에서 넣어주는것과, 효율적으로 짤라서 넣어주는것(chunk)에 따라 할루시네이션을 방지할 수 있다.
    3. Chunk와 유사도를 통해 필요한 chunk를 잡는데, 잘못나눈다면, 높게 유사도검색이 안나와서 답변이 재대로 할 수 없다.
    4. 따라서, 다양한 전략을 통해 쓸대없는 정보를 제고하고 필요한 정보를 주어야한다.

- 다양항 문서 분할 과정
    1. 문서 구조 파악
    2. 단위 선정
    3. 단위 크기선정(chunk size)
    4. 청크오버랩(chunk overlap)

## 02. CharcterTextSpliter

이 방법은 가장 간단한 방식입니다.

기본적으로 `"\n\n"` 을 기준으로 문자 단위로 텍스트를 분할하고, 청크의 크기를 문자 수로 측정합니다.

1. 텍스트 분할 방식: 단일 문자 기준
2. 청크 크기 측정 방식: 문자 수 기준


```python
# data/appendix-keywords.txt 파일을 열어서 f라는 파일 객체를 생성합니다.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # 파일의 내용을 읽어서 file 변수에 저장합니다.

# 파일으로부터 읽은 내용을 일부 출력합니다.
print(file[:500])
```

### CharacterTextSplitter를 사용하여 분할

CharacterTextSplitter를 사용하여 텍스트를 청크(chunk)로 분할하는 코드에 대해 설명합니다.

- `separator`: 분할할 기준을 설정합니다. 기본값은 `"\n\n"`입니다.
- `chunk_size`: 각 청크의 최대 크기를 설정합니다. 예: 250자
- `chunk_overlap`: 인접한 청크 간 중복을 허용합니다. 예: 50자
- `length_function`: 텍스트의 길이를 계산하는 함수를 지정합니다. 예: `len`

```python
from langchain_text_splitters import CharacterTextSplitter

# CharacterTextSplitter를 사용하여 텍스트를 청크(chunk)로 분할하는 코드
text_splitter = CharacterTextSplitter(
    # 텍스트를 분할할 때 사용할 구분자를 지정합니다. 기본값은 "\n\n"입니다.
    separator="\n\n",
    # 분할된 텍스트 청크의 최대 크기를 지정합니다 (문자 수).
    chunk_size=210,
    # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
    chunk_overlap=0,
    # 텍스트의 길이를 계산하는 함수를 지정합니다.
    length_function=len,
)

# 보통은 chunk_size, chunk_overlap을 많이 활용함
```

- `text_splitter`를 사용하여 `file` 텍스트를 문서 단위로 분할합니다.
- 분할된 문서 리스트 중 첫 번째 문서(`texts[0]`)를 출력합니다.

```python
# 텍스트를 청크로 분할합니다.
texts = text_splitter.create_documents([file])
print(len(texts[0].page_content))  # 분할된 문서의 개수를 출력합니다.
print(texts[0])  # 분할된 문서 중 첫 번째 문서를 출력합니다.
```

다음은 문서와 함께 메타데이터를 전달하는 예시입니다.

메타데이터가 문서와 함께 분할되는 점에 주목해 주세요.

- `create_documents` 메서드는 텍스트 데이터와 메타데이터 리스트를 인자로 받습니다.

```python
metadatas = [
    {"document": 1},
    {"document": 2},
]  # 문서에 대한 메타데이터 리스트를 정의합니다.
documents = text_splitter.create_documents(
    [
        file, # 첫번째 pdf
        file, # 두번째 pdf
    ],  # 분할할 텍스트 데이터를 리스트로 전달합니다.
    metadatas=metadatas,  # 각 문서에 해당하는 메타데이터를 전달합니다.
)
print(documents[0])  # 분할된 문서 중 첫 번째 문서를 출력합니다.
```

`split_text()` 메서드를 사용하여 텍스트를 분할합니다.

- text_spliter.split_documents([file])
- 넣어주는 파일이 다큐맨트면 위에같이 넣어준다.
- `text_splitter.split_text(file)[0]`은 `file` 텍스트를 `text_splitter`를 사용하여 분할한 후, 분할된 텍스트 조각 중 첫 번째 요소를 반환합니다.

## 03 RecursiveCharacterTextSplitter

이 텍스트 분할기는 일반적인 텍스트에 권장되는 방식입니다.

이 분할기는 문자 목록을 매개변수로 받아 동작합니다.

분할기는 청크가 충분히 작아질 때까지 주어진 문자 목록의 순서대로 텍스트를 분할하려고 시도합니다.
- 이 규칙만 바라보고 짜른다

기본 문자 목록은 `["\n\n", "\n", " ", ""]`입니다.
    - 이걸 기준으로 최대한 맞춰서
    - 순서도 중요하다! 순서에 맞춰서 split을 진행한다.

- **단락** -> **문장** -> **단어** 순서로 재귀적으로 분할합니다.

이는 단락(그 다음으로 문장, 단어) 단위가 의미적으로 가장 강하게 연관된 텍스트 조각으로 간주되므로, 가능한 한 함께 유지하려는 효과가 있습니다.

1. 텍스트가 분할되는 방식: 문자 목록(`["\n\n", "\n", " ", ""]`) 에 의해 분할됩니다.

2. 청크 크기가 측정되는 방식: 문자 수에 의해 측정됩니다.


- `appendix-keywords.txt` 파일을 열어 내용을 읽어들입니다.
- 읽어들인 내용을 `file` 변수에 저장합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
    chunk_size=250,
    # 청크 간의 중복되는 문자 수를 설정합니다.
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
)

```

`RecursiveCharacterTextSplitter`를 사용하여 텍스트를 작은 청크로 분할하는 예제입니다.

- `chunk_size`를 250 으로 설정하여 각 청크의 크기를 제한합니다.
- `chunk_overlap`을 50 으로 설정하여 인접한 청크 간에 50 개 문자의 중첩을 허용합니다.
- `length_function`으로 `len` 함수를 사용하여 텍스트의 길이를 계산합니다.
- `is_separator_regex`를 `False`로 설정하여 구분자로 정규식을 사용하지 않습니다.

```python
# text_splitter를 사용하여 file 텍스트를 문서로 분할합니다.
texts = text_splitter.create_documents([file])
print(texts[0])  # 분할된 문서의 첫 번째 문서를 출력합니다.
print("===" * 20)
print(texts[1])  # 분할된 문서의 두 번째 문서를 출력합니다.
```

- `text_splitter`를 사용하여 `file` 텍스트를 문서 단위로 분할합니다.
- 분할된 문서는 `texts` 리스트에 저장됩니다.
- `print(texts[0])`과 `print(texts[1])`을 통해 분할된 문서의 첫 번째와 두 번째 문서를 출력합니다.

```python
# 텍스트를 분할하고 분할된 텍스트의 처음 2개 요소를 반환합니다.
text_splitter.split_text(file)[:2]
```

- `text_splitter.split_text()` 함수를 사용하여 `file` 텍스트를 분할합니다.

## 04. TokenTextSplitter

언어 모델에는 토큰 제한이 있습니다. 따라서 토큰 제한을 초과하지 않아야 합니다.

`TokenTextSplitter` 는 텍스트를 토큰 수를 기반으로 청크를 생성할 때 유용합니다.

- 다양한 토큰화 방법이 있다.
- 다양한 토큰화 알고리즘에 따라 토큰이 달라지고 청크결과가 달라진다.

## tiktoken

`tiktoken` 은 OpenAI에서 만든 빠른 `BPE Tokenizer` 입니다.
    - tokenizer다시봐

- `./data/appendix-keywords.txt` 파일을 열어 내용을 읽어들입니다.
- 읽어들인 내용을 `file` 변수에 저장합니다.

```python
# data/appendix-keywords.txt 파일을 열어서 f라는 파일 객체를 생성합니다.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # 파일의 내용을 읽어서 file 변수에 저장합니다.

```
- 파일로부터 읽은 파일의 일부 내용을 출력합니다.

`CharacterTextSplitter`를 사용하여 텍스트를 분할합니다.

- `from_tiktoken_encoder` 메서드를 사용하여 Tiktoken 인코더 기반의 텍스트 분할기를 초기화합니다.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    # 청크 크기를 300으로 설정합니다.
    chunk_size=300,
    # 청크 간 중복되는 부분이 없도록 설정합니다.
    chunk_overlap=0,
)
# file 텍스트를 청크 단위로 분할합니다.
texts = text_splitter.split_text(file)
```

분할된 청크의 개수를 출력합니다.

**참고**

- `CharacterTextSplitter.from_tiktoken_encoder`를 사용하는 경우, 텍스트는 `CharacterTextSplitter`에 의해서만 분할되고 `tiktoken` 토크나이저는 분할된 텍스트를 병합하는 데 사용됩니다. (이는 분할된 텍스트가 `tiktoken` 토크나이저로 측정한 청크 크기보다 클 수 있음을 의미합니다.)

- `RecursiveCharacterTextSplitter.from_tiktoken_encoder`를 사용하면 분할된 텍스트가 언어 모델에서 허용하는 토큰의 청크 크기보다 크지 않도록 할 수 있으며, 각 분할은 크기가 더 큰 경우 재귀적으로 분할됩니다. 또한 tiktoken 분할기를 직접 로드할 수 있으며, 이는 각 분할이 청크 크기보다 작음을 보장합니다.


### TokenTextSplitter

- `TokenTextSplitter` 클래스를 사용하여 텍스트를 토큰 단위로 분할합니다.

```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=300,  # 청크 크기를 10으로 설정합니다.
    chunk_overlap=0,  # 청크 간 중복을 0으로 설정합니다.
)

# state_of_the_union 텍스트를 청크로 분할합니다.
texts = text_splitter.split_text(file)
print(texts[0])  # 분할된 텍스트의 첫 번째 청크를 출력합니다.
```
- 특수 문자가 나올 수 있다. 토큰단위이기떄문에
- BPE Tokenizer로 하기떄문에

### spaCy

spaCy는 Python과 Cython 프로그래밍 언어로 작성된 고급 자연어 처리를 위한 오픈 소스 소프트웨어 라이브러리입니다.

NLTK의 또 다른 대안은 spaCy tokenizer를 사용하는 것입니다.

1. 텍스트가 분할되는 방식: **spaCy tokenizer**에 의해 분할됩니다.

2. chunk size가 측정되는 방법: **문자 수**로 측정됩니다.

- `SpacyTextSplitter` 클래스를 사용하여 텍스트 분할기를 생성합니다.


### SentenceTransformers

`SentenceTransformersTokenTextSplitter`는 `sentence-transformer` 모델에 특화된 텍스트 분할기입니다.

기본 동작은 사용하고자 하는 sentence transformer 모델의 토큰 윈도우에 맞게 텍스트를 청크로 분할하는 것입니다.

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# 문장 분할기를 생성하고 청크 간 중복을 0으로 설정합니다.
splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=0)

# data/appendix-keywords.txt 파일을 열어서 f라는 파일 객체를 생성합니다.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # 파일의 내용을 읽어서 file 변수에 저장합니다.

# 파일으로부터 읽은 내용을 일부 출력합니다.
print(file[:350])

```


### NLTK

Natural Language Toolkit (NLTK)은 Python 프로그래밍 언어로 작성된 영어 자연어 처리(NLP)를 위한 라이브러리와 프로그램 모음입니다.

단순히 "\n\n"으로 분할하는 대신, NLTK tokenizers를 기반으로 텍스트를 분할하는 데 NLTK를 사용할 수 있습니다.

1. 텍스트 분할 방법: NLTK tokenizer에 의해 분할됩니다.
2. chunk 크기 측정 방법: 문자 수에 의해 측정됩니다.


- `nltk` 라이브러리를 설치하는 pip 명령어입니다.
- NLTK(Natural Language Toolkit)는 자연어 처리를 위한 파이썬 라이브러리입니다.
- 텍스트 데이터의 전처리, 토큰화, 형태소 분석, 품사 태깅 등 다양한 NLP 작업을 수행할 수 있습니다.


## KoNLPy

KoNLPy(Korean NLP in Python)는 한국어 자연어 처리(NLP)를 위한 파이썬 패키지입니다.

토큰 분할은 텍스트를 토큰이라고 하는 더 작고 관리하기 쉬운 단위로 분할하는 과정을 포함합니다.

이러한 토큰은 종종 단어, 구, 기호 또는 추가 처리 및 분석에 중요한 다른 의미 있는 요소입니다.

영어와 같은 언어에서 토큰 분할은 일반적으로 공백과 구두점으로 단어를 분리하는 것을 포함합니다.

토큰 분할의 효과는 언어 구조에 대한 토크나이저의 이해에 크게 의존하며, 이는 의미 있는 토큰 생성을 보장합니다.

영어를 위해 설계된 토크나이저는 한국어와 같은 다른 언어의 고유한 의미 구조를 이해할 수 있는 능력이 없기 때문에 한국어 처리에 효과적으로 사용될 수 없습니다.

## KoNLPy의 Kkma 분석기를 사용한 한국어 토큰 분할

한국어 텍스트의 경우 KoNLPY에는 `Kkma`(Korean Knowledge Morpheme Analyzer)라는 형태소 분석기가 포함되어 있습니다.

`Kkma`는 한국어 텍스트에 대한 상세한 형태소 분석을 제공합니다.

문장을 단어로, 단어를 각각의 형태소로 분해하고 각 토큰에 대한 품사를 식별합니다.

텍스트 블록을 개별 문장으로 분할할 수 있어 긴 텍스트 처리에 특히 유용합니다.

### 사용시 고려사항

`Kkma`는 상세한 분석으로 유명하지만, 이러한 정밀성이 처리 속도에 영향을 미칠 수 있다는 점에 유의해야 합니다. 따라서 `Kkma`는 신속한 텍스트 처리보다 분석적 깊이가 우선시되는 애플리케이션에 가장 적합합니다.


- KoNLPy 라이브러리를 설치하는 pip 명령어입니다.
- KoNLPy는 한국어 자연어 처리를 위한 파이썬 패키지로, 형태소 분석, 품사 태깅, 구문 분석 등의 기능을 제공합니다.


## Hugging Face tokenizer

Hugging Face는 다양한 토크나이저를 제공합니다.

이 코드에서는 Hugging Face의 토크나이저 중 하나인 GPT2TokenizerFast를 사용하여 텍스트의 토큰 길이를 계산합니다.

텍스트 분할 방식은 다음과 같습니다:

- 전달된 문자 단위로 분할됩니다.

청크 크기 측정 방식은 다음과 같습니다:

- Hugging Face 토크나이저에 의해 계산된 토큰 수를 기준으로 합니다.

- `GPT2TokenizerFast` 클래스를 사용하여 `tokenizer` 객체를 생성합니다.
- `from_pretrained` 메서드를 호출하여 사전 학습된 "gpt2" 토크나이저 모델을 로드합니다.


`from_huggingface_tokenizer` 메서드를 통해 허깅페이스 토크나이저(`tokenizer`)를 사용하여 텍스트 분할기를 초기화합니다.

1. Hugging Face tokenizer 다양한것들이  많다
2. KONLP 한국어 토큰에 좋다


## 05. SemanticChunker

텍스트를 의미론적 유사성에 기반하여 분할합니다.

**Reference**

- [Greg Kamradt의 노트북](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

이 방법은 텍스트를 문장 단위로 분할한 후, 3개의 문장씩 그룹화하고, 임베딩 공간에서 유사한 문장들을 병합하는 과정을 거칩니다.

### SemanticChunker 생성

`SemanticChunker`는 LangChain의 실험적 기능 중 하나로, 텍스트를 의미론적으로 유사한 청크로 분할하는 역할을 합니다.

이를 통해 텍스트 데이터를 보다 효과적으로 처리하고 분석할 수 있습니다.

`SemanticChunker`를 사용하여 텍스트를 의미적으로 관련된 청크로 분할합니다.

```python

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# OpenAI 임베딩을 사용하여 의미론적 청크 분할기를 초기화합니다.
text_splitter = SemanticChunker(OpenAIEmbeddings())

```

### 텍스트 분할
- `text_splitter`를 사용하여 `file` 텍스트를 문서 단위로 분할합니다.
- `create_documents()` 함수를 사용하여 청크를 문서로 변환할 수 있습니다.
- embeder만 잘 넣어 주면 된다.

```python
# text_splitter를 사용하여 분할합니다.
docs = text_splitter.create_documents([file])
print(docs[0].page_content)  # 분할된 문서 중 첫 번째 문서의 내용을 출력합니다.
```

## Breakpoints

이 chunker는 문장을 "분리"할 시점을 결정하여 작동합니다. 이는 두 문장 간의 임베딩 차이를 살펴봄으로써 이루어집니다.

그 차이가 특정 임계값을 넘으면 문장이 분리됩니다.

- 참고 영상: https://youtu.be/8OJC21T2SL4?si=PzUtNGYJ_KULq3-w&t=2580
- 너무 길게 짤린걸 breakingpoints로 짜를 수 있다.


### Percentile

기본적인 분리 방식은 백분위수(`Percentile`) 를 기반으로 합니다.

이 방법에서는 문장 간의 모든 차이를 계산한 다음, 지정한 백분위수를 기준으로 분리합니다.

```python
text_splitter = SemanticChunker(
    # OpenAI의 임베딩 모델을 사용하여 시맨틱 청커를 초기화합니다.
    OpenAIEmbeddings(),
    # 분할 기준점 유형을 백분위수로 설정합니다.
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

### Standard Deviation

이 방법에서는 지정한 `breakpoint_threshold_amount` 표준편차보다 큰 차이가 있는 경우 분할됩니다.

- `breakpoint_threshold_type` 매개변수를 "standard_deviation"으로 설정하여 청크 분할 기준을 표준편차 기반으로 지정합니다.

```python
text_splitter = SemanticChunker(
    # OpenAI의 임베딩 모델을 사용하여 시맨틱 청커를 초기화합니다.
    OpenAIEmbeddings(),
    # 분할 기준으로 표준 편차를 사용합니다.
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)
```

### Interquartile

이 방법에서는 사분위수 범위(interquartile range)를 사용하여 청크를 분할합니다.

- `breakpoint_threshold_type` 매개변수를 "interquartile"로 설정하여 청크 분할 기준을 사분위수 범위로 지정합니다.

```python
text_splitter = SemanticChunker(
    # OpenAI의 임베딩 모델을 사용하여 의미론적 청크 분할기를 초기화합니다.
    OpenAIEmbeddings(),
    # 분할 기준점 임계값 유형을 사분위수 범위로 설정합니다.
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=0.5,
)

# text_splitter를 사용하여 분할합니다.
docs = text_splitter.create_documents([file])

# 결과를 출력합니다.
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(doc.page_content)  # 분할된 문서 중 첫 번째 문서의 내용을 출력합니다.
    print("===" * 20)

print(len(docs))  # docs의 길이를 출력합니다.
```
- 직접 실험을 통해 다 짤라보고 의미적으로 잘 맞는것을 확인한다.


## 06. codespliter

- code기반 RAG를 만들떄 필요함

### Split code

CodeTextSplitter를 사용하면 다양한 프로그래밍 언어로 작성된 코드를 분할할 수 있습니다.

이를 위해서는 `Language` enum을 import하고, 해당하는 프로그래밍 언어를 지정해주면 됩니다.


`RecursiveCharacterTextSplitter`를 사용하여 텍스트를 분할하는 예제입니다.

- `langchain_text_splitters` 모듈에서 `Language`와 `RecursiveCharacterTextSplitter` 클래스를 임포트합니다.
- `RecursiveCharacterTextSplitter`는 텍스트를 문자 단위로 재귀적으로 분할하는 텍스트 분할기입니다.

```python
# 지원되는 언어의 전체 목록을 가져옵니다.
[e.value for e in Language]
```

`RecursiveCharacterTextSplitter` 클래스의 `get_separators_for_language` 메서드를 사용하여 특정 언어에 사용되는 구분자(separators)를 확인할 수 있습니다.

- 예시에서는 `Language.PYTHON` 열거형 값을 인자로 전달하여 Python 언어에 사용되는 구분자를 확인합니다.


### Python

`RecursiveCharacterTextSplitter` 사용한 예제는 다음과 같습니다.

- `RecursiveCharacterTextSplitter`를 사용하여 Python 코드를 문서 단위로 분할합니다.
  - `language` 매개변수에 `Language.PYTHON`을 지정하여 Python 언어를 사용합니다.
  - `chunk_size`를 50으로 설정하여 각 문서의 최대 크기를 제한합니다.
  - `chunk_overlap`을 0으로 설정하여 문서 간의 중복을 허용하지 않습니다.

```python
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

hello_world()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
```

- `Document` 를 생성합니다. 생성된 `Document` 는 리스트 형태로 반환됩니다.

```python
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs


for doc in python_docs:
    print(doc.page_content, end="\n==================\n")
```

### Markdown

다음은 Markdown 텍스트 분할기를 사용한 예시입니다.


## 07. MarkdownHeaderTextSplitter

- PDF인식이 안될 경우 -> MarkDown으로 변환해서 사용하는 경우가 많다.

마크다운 파일의 구조를 이해하고 효율적으로 다루는 것은 문서 작업에 있어 매우 중요할 수 있습니다. 특히, 문서의 전체적인 맥락과 구조를 고려하여 의미 있는 방식으로 텍스트를 임베딩하는 과정은, 광범위한 의미와 주제를 더 잘 포착할 수 있는 포괄적인 벡터 표현을 생성하는 데 큰 도움이 됩니다.

이러한 맥락에서, 마크다운 파일의 특정 부분, 즉 헤더별로 내용을 나누고 싶을 때가 있습니다. 예를 들어, 문서 내에서 각각의 헤더 아래에 있는 내용을 기반으로 서로 연관된 정보 덩어리, 즉 '청크'를 만들고 싶은 경우가 그러합니다. 이는 텍스트의 공통된 맥락을 유지하면서도, 문서의 구조적 요소를 효과적으로 활용하려는 시도입니다.

이런 과제를 해결하기 위해, `MarkdownHeaderTextSplitter` 라는 도구를 활용할 수 있습니다. 이 도구는 문서를 지정된 헤더 집합에 따라 분할하여, 각 헤더 그룹 아래의 내용을 별도의 청크로 관리할 수 있게 합니다. 이 방법을 통해, 문서의 전반적인 구조를 유지하면서도 내용을 더 세밀하게 다룰 수 있게 되며, 이는 다양한 처리 과정에서 유용하게 활용될 수 있습니다.


`MarkdownHeaderTextSplitter`를 사용하여 마크다운 형식의 텍스트를 헤더 단위로 분할합니다.

- 마크다운 문서의 헤더(`#`, `##`, `###` 등)를 기준으로 텍스트를 분할하는 역할을 합니다.

- `markdown_document` 변수에 마크다운 형식의 문서가 할당됩니다.
- `headers_to_split_on` 리스트에는 마크다운 헤더 레벨과 해당 레벨의 이름이 튜플 형태로 정의됩니다.
- `MarkdownHeaderTextSplitter` 클래스를 사용하여 `markdown_splitter` 객체를 생성하며, `headers_to_split_on` 매개변수로 분할 기준이 되는 헤더 레벨을 전달합니다.
- `split_text` 메서드를 호출하여 `markdown_document`를 헤더 레벨에 따라 분할합니다.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운 형식의 문서를 문자열로 정의합니다.
markdown_document = "# Title\n\n## 1. SubTitle\n\nHi this is Jim\n\nHi this is Joe\n\n### 1-1. Sub-SubTitle \n\nHi this is Lance \n\n## 2. Baz\n\nHi this is Molly"
print(markdown_document)


headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
    (
        "#",
        "Header 1",
    ),  # 헤더 레벨 1은 '#'로 표시되며, 'Header 1'이라는 이름을 가집니다.
    (
        "##",
        "Header 2",
    ),  # 헤더 레벨 2는 '##'로 표시되며, 'Header 2'라는 이름을 가집니다.
    (
        "###",
        "Header 3",
    ),  # 헤더 레벨 3은 '###'로 표시되며, 'Header 3'이라는 이름을 가집니다.
]

# 마크다운 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```


기본적으로 `MarkdownHeaderTextSplitter`는 분할되는 헤더를 출력 청크의 내용에서 제거합니다.

이는 `strip_headers = False`로 설정하여 비활성화할 수 있습니다.

```python
markdown_splitter = MarkdownHeaderTextSplitter(
    # 분할할 헤더를 지정합니다.
    headers_to_split_on=headers_to_split_on,
    # 헤더를 제거하지 않도록 설정합니다.
    strip_headers=False,
)
# 마크다운 문서를 헤더를 기준으로 분할합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

각 마크다운 그룹 내에서는 원하는 텍스트 분할기(text splitter)를 적용할 수 있습니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

markdown_document = "# Intro \n\n## History \n\nMarkdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\nMarkdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n## Rise and divergence \n\nAs Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\nadditional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n#### Standardization \n\nFrom 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n# Implementations \n\nImplementations of Markdown are available for over a dozen programming languages."
print(markdown_document)


```

먼저, `MarkdownHeaderTextSplitter` 사용하여 마크다운 문서를 헤더를 기준으로 분할합니다.

```python
headers_to_split_on = [
    ("#", "Header 1"),  # 분할할 헤더 레벨과 해당 레벨의 이름을 지정합니다.
    # ("##", "Header 2"),
]

# Markdown 문서를 헤더 레벨에 따라 분할합니다.
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

이전의 `MarkdownHeaderTextSplitter` 로 분할된 결과를 다시 `RecursiveCharacterTextSplitter` 로 분할합니다.


```python
chunk_size = 200  # 분할된 청크의 크기를 지정합니다.
chunk_overlap = 20  # 분할된 청크 간의 중복되는 문자 수를 지정합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# 문서를 문자 단위로 분할합니다.
splits = text_splitter.split_documents(md_header_splits)
# 분할된 결과를 출력합니다.
for header in splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

## 08. HTMLHeaderTextSplitter

<a href="https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata">`MarkdownHeaderTextSplitter`</a>와 개념적으로 유사한 `HTMLHeaderTextSplitter`는 텍스트를 요소 수준에서 분할하고 각 헤더에 대한 메타데이터를 추가하는 "구조 인식" 청크 생성기입니다.

이는 각 청크와 "관련된" 메타데이터를 추가합니다.

`HTMLHeaderTextSplitter`는 요소별로 청크를 반환하거나 동일한 메타데이터를 가진 요소를 결합할 수 있으며

- (a) 관련 텍스트를 의미론적으로 (대략적으로) 그룹화하고
- (b) 문서 구조에 인코딩된 컨텍스트 풍부한 정보를 보존하는 것을 목표로 합니다.
- 마크다운과 비슷하지만, 재귀적인 부분이 들어간다.

## HTML 문자열을 사용하는 경우

- `headers_to_split_on` 리스트에 분할 기준이 되는 헤더 태그와 해당 헤더의 이름을 튜플 형태로 지정합니다.
- `HTMLHeaderTextSplitter` 객체를 생성하면서 `headers_to_split_on` 매개변수에 분할 기준 헤더 리스트를 전달합니다.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>헤더1</h1>
        <p>헤더1 에 포함된 본문</p>
        <div>
            <h2>헤더2-1 제목</h2>
            <p>헤더2-1 에 포함된 본문</p>
            <h3>헤더3-1 제목</h3>
            <p>헤더3-1 에 포함된 본문</p>
            <h3>헤더3-2 제목</h3>
            <p>헤더3-2 에 포함된 본문</p>
        </div>
        <div>
            <h2>헤더2-2 제목2</h2>
            <p>헤더2-2 에 포함된 본문</p>
        </div>
        <br>
        <p>마지막 내용</p>
    </div>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# 지정된 헤더를 기준으로 HTML 텍스트를 분할하는 HTMLHeaderTextSplitter 객체를 생성합니다.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# HTML 문자열을 분할하여 결과를 html_header_splits 변수에 저장합니다.
html_header_splits = html_splitter.split_text(html_string)
# 분할된 결과를 출력합니다.
for header in html_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

### 다른 splitter와 파이프라인으로 연결하고, 웹 URL에서 HTML을 로드하는 경우입니다.

이 예시에서는 웹 URL로부터 HTML 콘텐츠를 로드한 후, 이를 다른 splitter와 파이프라인으로 연결하여 처리하는 과정입니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

url = "https://plato.stanford.edu/entries/goedel/"  # 분할할 텍스트의 URL을 지정합니다.

headers_to_split_on = [  # 분할할 HTML 헤더 태그와 해당 헤더의 이름을 지정합니다.
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

# HTML 헤더를 기준으로 텍스트를 분할하는 HTMLHeaderTextSplitter 객체를 생성합니다.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# URL에서 텍스트를 가져와 HTML 헤더를 기준으로 분할합니다.
html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 500  # 텍스트를 분할할 청크의 크기를 지정합니다.
chunk_overlap = 30  # 분할된 청크 간의 중복되는 문자 수를 지정합니다.
text_splitter = RecursiveCharacterTextSplitter(  # 텍스트를 재귀적으로 분할하는 RecursiveCharacterTextSplitter 객체를 생성합니다.
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# HTML 헤더로 분할된 텍스트를 다시 청크 크기에 맞게 분할합니다.
splits = text_splitter.split_documents(html_header_splits)

# 분할된 텍스트 중 80번째부터 85번째까지의 청크를 출력합니다.
for header in splits[80:85]:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

### 한계

HTMLHeaderTextSplitter는 HTML 문서 간의 구조적 차이를 처리하려고 시도하지만, 때로는 특정 헤더를 누락할 수 있습니다.

예를 들어, 이 알고리즘은 헤더가 항상 관련 텍스트보다 "위"에 있는 노드, 즉 이전 형제 노드, 조상 노드 및 이들의 조합에 위치한다고 가정합니다.

다음 뉴스 기사(이 문서 작성 시점 기준)에서는 최상위 헤드라인의 텍스트가 "h1"으로 태그되어 있지만, 우리가 예상하는 텍스트 요소와는 **별개의 하위 트리** 에 있는 것을 볼 수 있습니다.

따라서 "h1" 요소와 관련 텍스트는 청크 메타데이터에 나타나지 않지만, 해당되는 경우 "h2"와 관련 텍스트는 볼 수 있습니다.

```python
# 분할할 HTML 페이지의 URL을 지정합니다.
url = "https://www.cnn.com/2023/09/25/weather/el-nino-winter-us-climate/index.html"

headers_to_split_on = [
    ("h1", "Header 1"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
    ("h2", "Header 2"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
]

# 지정된 헤더를 기준으로 HTML 텍스트를 분할하는 HTMLHeaderTextSplitter 객체를 생성합니다.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 지정된 URL의 HTML 페이지를 분할하여 결과를 html_header_splits 변수에 저장합니다.
html_header_splits = html_splitter.split_text_from_url(url)

# 분할된 결과를 출력합니다.
for header in html_header_splits:
    print(f"{header.page_content[:100]}")
    print(f"{header.metadata}", end="\n=====================\n")
```