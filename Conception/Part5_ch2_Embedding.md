# Ch5_RAG_ch2 임베딩(Embeding)

1. Load
2. Split
3. EMBED
    - 차원이 다름
    - 차원이 높으면 상세하게 나타낼 수 있으나, 실제로는 높다고 무조건 성능이 좋다고 할 수 없음
    - 종류 : OpneAI Embeding, huggingfaceEmbedding, UpstageEmbedding(한국어에 특화가 잘되어있음)
4. STORE

- pdf참고
- 임베딩의 필요성
    - 의미 이해 : 임베딩을 통해 이러한 텍스트를 정량화된 형태로 변환함으로써, 컴퓨터가 문서 내용의미를 더 잘 이해할 수 있도록 처리
    - 청보 검색 향성 :수치화된 벡터 형태로의 변환은 문서간의 유사성을 계산하는데 있어 필수적이다. 이는 관련 문서를 검색하거나 질문에 가장 적합한 문서를 찾는 작업을 용이하게 해준다.
- 임베딩을 통해 유사도 계산을 진행
- openAI
    - 다국어 성능이 좋다
    - 하드웨어 resource가 많이 들기 때문에, 계산이 많이 들지만, openAI를 통해 비용이 발생하게 된다.
- Huggingface
    - 한국어 임베딩 성능이 좋은 BGE-M3 / e5를 무료로만들어서 사용할 수 있다.

- Ollama에서 제공해주는 임베딩 제공
    - 무료 Opensource를 사용한다.

- 임베딩의 유사도계산

## 2.OpenAIEmbeddings

문서 임베딩은 문서의 내용을 수치적인 벡터로 변환하는 과정입니다. 

이 과정을 통해 문서의 의미를 수치화하고, 다양한 자연어 처리 작업에 활용할 수 있습니다. 대표적인 사전 학습된 언어 모델로는 BERT와 GPT가 있으며, 이러한 모델들은 문맥적 정보를 포착하여 문서의 의미를 인코딩합니다. 

문서 임베딩은 토큰화된 문서를 모델에 입력하여 임베딩 벡터를 생성하고, 이를 평균하여 전체 문서의 벡터를 생성합니다. 이 벡터는 문서 분류, 감성 분석, 문서 간 유사도 계산 등에 활용될 수 있습니다.

[더 알아보기](https://platform.openai.com/docs/guides/embeddings/embedding-models)

## 설정

먼저 langchain-openai를 설치하고 필요한 환경 변수를 설정합니다.

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH08-Embeddings")

```

지원되는 모델 목록

| MODEL                  | PAGES PER DOLLAR | PERFORMANCE ON MTEB EVAL | MAX INPUT |
|------------------------|------------------|---------------------------|-----------|
| text-embedding-3-small | 62,500           | 62.3%                     | 8191      |
| text-embedding-3-large | 9,615            | 64.6%                     | 8191      |
| text-embedding-ada-002 | 12,500           | 61.0%                     | 8191      |

- ada 모델을 가장 많이 씀 / default모델 이기도함
- 스몰모델이 더좋음...

```python

from langchain_openai import OpenAIEmbeddings

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### 쿼리 임베딩

`embeddings.embed_query(text)`는 주어진 텍스트를 임베딩 벡터로 변환하는 함수입니다.

이 함수는 텍스트를 벡터 공간에 매핑하여 의미적으로 유사한 텍스트를 찾거나 텍스트 간의 유사도를 계산하는 데 사용될 수 있습니다.

`query_result[:5]`는 `query_result` 리스트의 처음 5개 요소를 슬라이싱(slicing)하여 선택합니다.

```python
# 텍스트를 임베딩하여 쿼리 결과를 생성합니다.
query_result = embeddings.embed_query(text)

# 쿼리 결과의 처음 5개 항목을 선택합니다.
query_result[:5]

```
### Document 임베딩

- 문서를 청킹 4개
- 1번문단, 2번문단 ..., 4문단
- [4, 1536]
- 통으로 넣어준다.


`embeddings.embed_documents()` 함수를 사용하여 텍스트 문서를 임베딩합니다.

- `[text]`를 인자로 전달하여 단일 문서를 리스트 형태로 임베딩 함수에 전달합니다.
- 함수 호출 결과로 반환된 임베딩 벡터를 `doc_result` 변수에 할당합니다.



```python
doc_result = embeddings.embed_documents(
    [text, text, text, text]
)  # 텍스트를 임베딩하여 문서 벡터를 생성합니다.
```

### 차원 지정

`text-embedding-3` 모델 클래스를 사용하면 반환되는 임베딩의 크기를 지정할 수 있습니다.

예를 들어, 기본적으로 `text-embedding-3-small`는 1536 차원의 임베딩을 반환합니다.


- Vector DB를 쓰는데 차원 변환이 필요할 수 있다. DB마다 차원이 다르기 떄문에(1024 차원일때 dim을 바꿔야한다.)
```python
# OpenAI의 "text-embedding-3-small" 모델을 사용하여 1024차원의 임베딩을 생성하는 객체를 초기화합니다.
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
# 주어진 텍스트를 임베딩하고 첫 번째 임베딩 벡터의 길이를 반환합니다.
len(embeddings_1024.embed_documents([text])[0])
```
### 유사도계산

```python
from sklearn.metrics.pairwise import cosine_similarity

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5]
embedded_sentences = embeddings_1024.embed_documents(sentences)

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

# sentence1 = "안녕하세요? 반갑습니다."
# sentence2 = "안녕하세요? 만나서 반가워요."
# sentence3 = "Hi, nice to meet you."
# sentence4 = "I like to eat apples."

for i, sentence in enumerate(embedded_sentences):
    for j, other_sentence in enumerate(embedded_sentences):
        if i < j:
            print(
                f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
            )
```

## 03.CacheBackedEmbeddings

Embeddings는 재계산을 피하기 위해 저장되거나 일시적으로 캐시될 수 있습니다.

Embeddings를 캐싱하는 것은 `CacheBackedEmbeddings`를 사용하여 수행될 수 있습니다. 캐시 지원 embedder는 embeddings를 키-값 저장소에 캐싱하는 embedder 주변에 래퍼입니다. 텍스트는 해시되고 이 해시는 캐시에서 키로 사용됩니다.

`CacheBackedEmbeddings`를 초기화하는 주요 지원 방법은 `from_bytes_store`입니다. 이는 다음 매개변수를 받습니다:

- `underlying_embeddings`: 임베딩을 위해 사용되는 embedder.
- `document_embedding_cache`: 문서 임베딩을 캐싱하기 위한 `ByteStore` 중 하나.
- `namespace`: (선택 사항, 기본값은 `""`) 문서 캐시를 위해 사용되는 네임스페이스. 이 네임스페이스는 다른 캐시와의 충돌을 피하기 위해 사용됩니다. 예를 들어, 사용된 임베딩 모델의 이름으로 설정하세요.

**주의**: 동일한 텍스트가 다른 임베딩 모델을 사용하여 임베딩될 때 충돌을 피하기 위해 `namespace` 매개변수를 설정하는 것이 중요합니다.




### LocalFileStore 에서 임베딩 사용 (영구 보관)

먼저, 로컬 파일 시스템을 사용하여 임베딩을 저장하고 FAISS 벡터 스토어를 사용하여 검색하는 예제를 살펴보겠습니다.

```python
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# OpenAI 임베딩을 사용하여 기본 임베딩 설정
embedding = OpenAIEmbeddings()

# 로컬 파일 저장소 설정
store = LocalFileStore("./cache/")

# 캐시를 지원하는 임베딩 생성
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
)

# store에서 키들을 순차적으로 가져옵니다.
list(store.yield_keys())


from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 문서 로드
raw_documents = TextLoader("./data/appendix-keywords.txt").load()
# 문자 단위로 텍스트 분할 설정
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 문서 분할
documents = text_splitter.split_documents(raw_documents)

# 코드 실행 시간을 측정합니다.
%time db = FAISS.from_documents(documents, cached_embedder)  # 문서로부터 FAISS 데이터베이스 생성
```

벡터 저장소를 다시 생성하려고 하면, 임베딩을 다시 계산할 필요가 없기 때문에 훨씬 더 빠르게 처리됩니다.

```python
# 캐싱된 임베딩을 사용하여 FAISS 데이터베이스 생성
%time db2 = FAISS.from_documents(documents, cached_embedder)
```

### `InmemoryByteStore` 사용 (비영구적)

다른 `ByteStore`를 사용하기 위해서는 `CacheBackedEmbeddings`를 생성할 때 해당 `ByteStore`를 사용하면 됩니다.

아래에서는, 비영구적인 `InMemoryByteStore`를 사용하여 동일한 캐시된 임베딩 객체를 생성하는 예시를 보여줍니다.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

store = InMemoryByteStore()  # 메모리 내 바이트 저장소 생성

# 캐시 지원 임베딩 생성
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding, store, namespace=embedding.model
)
```

## 04. 로컬 모델 임베딩(HuggingFaceEmbedding)

### HuggingFace Endpoint Embedding

- `HuggingFaceEndpointEmbeddings` 는 내부적으로 InferenceClient를 사용하여 임베딩을 계산한다는 점에서 HufcxgdgggingFaceEndpoint가 LLM에서 수행하는 것과 매우 유사합니다.

```python

```


```python

```


```python

```


```python

```