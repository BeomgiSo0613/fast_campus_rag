# VectorStore

- vectorstore pdf
- VectorDB : 저장하는곳
    - 검색 성능의 차이가 거의없지만, VectorDB마다 가지고있는 정보를 찾는 검색에 따라 결과가 달라지기도함
    - Keyword검색을 지원할 수도 있고 아닐수도 있음
    - Semantics는 왠만하면 가능함
    - local 일지/ cloud일지 선택해야함
    - local 돈이 안들어서 좋지만, 문서의 양이 늘어나면 확 느려지는 경우가 있다.
        - 확장성이 떨어진다.
    
- 벡터스토어의 필요성
    1.
    2.
    3.

벡터스토어의 중요성

- 벡터스토어 저장 단계는 RAG 시스템의 검색 기능과 직접적으로 연결되어 있으며, 전체 시스템의 응답 시간과 정확성에 큰 영향이 미친다. 이 단계를 통해 데이터가 잘 관리되고, 필요할 때 즉시 접근할 수 있도록 함으로써, 사용자에게 신속하고 정확한 정보를 제공할 수 있다.

# Chroma

이 노트북에서는 Chroma 벡터스토어를 시작하는 방법을 다룹니다.

Chroma는 개발자의 생산성과 행복에 초점을 맞춘 AI 네이티브 오픈 소스 벡터 데이터베이스입니다. Chroma는 Apache 2.0에 따라 라이선스가 부여됩니다. 


**참고링크**

- [Chroma LangChain 문서](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/)
- [Chroma 공식문서](https://docs.trychroma.com/getting-started)
- [LangChain 지원 VectorStore 리스트](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)

## VectorStore 생성

- 전체적인 구조가 비슷하다.
- 나의상황에 잘 맞게 써야한다
- local에서는 chroma가 깡패다


### 벡터 저장소 생성 (from_documents)

`from_documents` 클래스 메서드는 문서 리스트로부터 벡터 저장소를 생성합니다. 

**매개변수**

- `documents` (List[Document]): 벡터 저장소에 추가할 문서 리스트
- `embedding` (Optional[Embeddings]): 임베딩 함수. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None
- `collection_name` (str): 생성할 컬렉션 이름.
- `persist_directory` (Optional[str]): 컬렉션을 저장할 디렉토리. 기본값은 None
- `client_settings` (Optional[chromadb.config.Settings]): Chroma 클라이언트 설정
- `client` (Optional[chromadb.Client]): Chroma 클라이언트 인스턴스
- `collection_metadata` (Optional[Dict]): 컬렉션 구성 정보. 기본값은 None

**참고**

- `persist_directory`가 지정되면 컬렉션이 해당 디렉토리에 저장됩니다. 지정되지 않으면 데이터는 메모리에 임시로 저장됩니다.
- 이 메서드는 내부적으로 `from_texts` 메서드를 호출하여 벡터 저장소를 생성합니다.
- 문서의 `page_content`는 텍스트로, `metadata`는 메타데이터로 사용됩니다.

**반환값**

- `Chroma`: 생성된 Chroma 벡터 저장소 인스턴스

생성시 `documents` 매개변수로 `Document` 리스트를 전달합니다. embedding 에 활용할 임베딩 모델을 지정하며, `namespace` 의 역할을 하는 `collection_name` 을 지정할 수 있습니다.


`persist_directory` 지정시 disk 에 파일 형태로 저장합니다.

### 벡터 저장소에 문서 추가

`add_documents` 메서드는 벡터 저장소에 문서를 추가하거나 업데이트합니다.

**매개변수**

- `documents` (List[Document]): 벡터 저장소에 추가할 문서 리스트
- `**kwargs`: 추가 키워드 인자
  - `ids`: 문서 ID 리스트 (제공 시 문서의 ID보다 우선함)

**참고**

- `add_texts` 메서드가 구현되어 있어야 합니다.
- 문서의 `page_content`는 텍스트로, `metadata`는 메타데이터로 사용됩니다.
- 문서에 ID가 있고 `kwargs`에 ID가 제공되지 않으면 문서의 ID가 사용됩니다.
- `kwargs`의 ID와 문서 수가 일치하지 않으면 ValueError가 발생합니다.

**반환값**

- `List[str]`: 추가된 텍스트의 ID 리스트

**예외**

- `NotImplementedError`: `add_texts` 메서드가 구현되지 않은 경우 발생

`add_texts` 메서드는 텍스트를 임베딩하고 벡터 저장소에 추가합니다.

**매개변수**

- `texts` (Iterable[str]): 벡터 저장소에 추가할 텍스트 리스트
- `metadatas` (Optional[List[dict]]): 메타데이터 리스트. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None

**참고**

- `ids`가 제공되지 않으면 UUID를 사용하여 자동으로 생성됩니다.
- 임베딩 함수가 설정되어 있으면 텍스트를 임베딩합니다.
- 메타데이터가 제공된 경우:
  - 메타데이터가 있는 텍스트와 없는 텍스트를 분리하여 처리합니다.
  - 메타데이터가 없는 텍스트의 경우 빈 딕셔너리로 채웁니다.
- 컬렉션에 upsert 작업을 수행하여 텍스트, 임베딩, 메타데이터를 추가합니다.

**반환값**

- `List[str]`: 추가된 텍스트의 ID 리스트

**예외**

- `ValueError`: 복잡한 메타데이터로 인한 오류 발생 시, 필터링 방법 안내 메시지와 함께 발생

기존의 아이디에 추가하는 경우 `upsert` 가 수행되며, 기존의 문서는 대체됩니다.


### 벡터 저장소에서 문서 삭제

`delete` 메서드는 벡터 저장소에서 지정된 ID의 문서를 삭제합니다.

**매개변수**

- `ids` (Optional[List[str]]): 삭제할 문서의 ID 리스트. 기본값은 None

**참고**

- 이 메서드는 내부적으로 컬렉션의 `delete` 메서드를 호출합니다.
- `ids`가 None이면 아무 작업도 수행하지 않습니다.

**반환값**

- None

### 초기화(reset_collection)

`reset_collection` 메서드는 벡터 저장소의 컬렉션을 초기화합니다.


### 벡터 저장소를 검색기(Retriever)로 변환

`as_retriever` 메서드는 벡터 저장소를 기반으로 VectorStoreRetriever를 생성합니다.

**매개변수**

- `**kwargs`: 검색 함수에 전달할 키워드 인자
  - `search_type` (Optional[str]): 검색 유형 (`"similarity"`, `"mmr"`, `"similarity_score_threshold"`)
  - `search_kwargs` (Optional[Dict]): 검색 함수에 전달할 추가 인자
    - `k`: 반환할 문서 수 (기본값: 4)
    - `score_threshold`: 최소 유사도 임계값
    - `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
    - `lambda_mult`: MMR 결과의 다양성 조절 (0~1, 기본값: 0.5)
    - `filter`: 문서 메타데이터 필터링

**반환값**

- `VectorStoreRetriever`: 벡터 저장소 기반 검색기 인스턴스

# 02.FAISS

Facebook AI Similarity Search (Faiss)는 밀집 벡터의 효율적인 유사도 검색과 클러스터링을 위한 라이브러리입니다.

Faiss는 RAM에 맞지 않을 수도 있는 벡터 집합을 포함하여 모든 크기의 벡터 집합을 검색하는 알고리즘을 포함하고 있습니다.

또한 평가와 매개변수 튜닝을 위한 지원 코드도 포함되어 있습니다.

**참고**
- [LangChain FAISS 문서](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/)
- [FAISS 문서](https://faiss.ai/)


## VectorStore 생성


**주요 초기화 매개변수**

1. 인덱싱 매개변수:
   - `embedding_function` (Embeddings): 사용할 임베딩 함수

2. 클라이언트 매개변수:
   - `index` (Any): 사용할 FAISS 인덱스
   - `docstore` (Docstore): 사용할 문서 저장소
   - `index_to_docstore_id` (Dict[int, str]): 인덱스에서 문서 저장소 ID로의 매핑

**참고**

- FAISS는 고성능 벡터 검색 및 클러스터링을 위한 라이브러리입니다.
- 이 클래스는 FAISS를 LangChain의 VectorStore 인터페이스와 통합합니다.
- 임베딩 함수, FAISS 인덱스, 문서 저장소를 조합하여 효율적인 벡터 검색 시스템을 구축할 수 있습니다.

- dim size를 확인해서 사용해야함
- 어떤 embeddings 모델을 사용하냐에따라 dim이 달라진


### FAISS 벡터 저장소 생성 (from_documents)

`from_documents` 클래스 메서드는 문서 리스트와 임베딩 함수를 사용하여 FAISS 벡터 저장소를 생성합니다.

**매개변수**

- `documents` (List[Document]): 벡터 저장소에 추가할 문서 리스트
- `embedding` (Embeddings): 사용할 임베딩 함수
- `**kwargs`: 추가 키워드 인자

**동작 방식**

1. 문서 리스트에서 텍스트 내용(`page_content`)과 메타데이터를 추출합니다.
2. 추출한 텍스트와 메타데이터를 사용하여 `from_texts` 메서드를 호출합니다.

**반환값**

- `VectorStore`: 문서와 임베딩으로 초기화된 벡터 저장소 인스턴스

**참고**

- 이 메서드는 `from_texts` 메서드를 내부적으로 호출하여 벡터 저장소를 생성합니다.
- 문서의 `page_content`는 텍스트로, `metadata`는 메타데이터로 사용됩니다.
- 추가적인 설정이 필요한 경우 `kwargs`를 통해 전달할 수 있습니다.


### FAISS 벡터 저장소 생성 (from_texts)

`from_texts` 클래스 메서드는 텍스트 리스트와 임베딩 함수를 사용하여 FAISS 벡터 저장소를 생성합니다.

**매개변수**

- `texts` (List[str]): 벡터 저장소에 추가할 텍스트 리스트
- `embedding` (Embeddings): 사용할 임베딩 함수
- `metadatas` (Optional[List[dict]]): 메타데이터 리스트. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None
- `**kwargs`: 추가 키워드 인자

**동작 방식**

1. 제공된 임베딩 함수를 사용하여 텍스트를 임베딩합니다.
2. 임베딩된 벡터와 함께 `__from` 메서드를 호출하여 FAISS 인스턴스를 생성합니다.

**반환값**

- `FAISS`: 생성된 FAISS 벡터 저장소 인스턴스

**참고**

- 이 메서드는 사용자 친화적인 인터페이스로, 문서 임베딩, 메모리 내 문서 저장소 생성, FAISS 데이터베이스 초기화를 한 번에 처리합니다.
- 빠르게 시작하기 위한 편리한 방법입니다.

**주의사항**

- 대량의 텍스트를 처리할 때는 메모리 사용량에 주의해야 합니다.
- 메타데이터나 ID를 사용하려면 텍스트 리스트와 동일한 길이의 리스트로 제공해야 합니다.


### 유사도 검색 (Similarity Search)

`similarity_search` 메서드는 주어진 쿼리와 가장 유사한 문서들을 검색하는 기능을 제공합니다.

**매개변수**

- `query` (str): 유사한 문서를 찾기 위한 검색 쿼리 텍스트
- `k` (int): 반환할 문서 수. 기본값은 4
- `filter` (Optional[Union[Callable, Dict[str, Any]]]): 메타데이터 필터링 함수 또는 딕셔너리. 기본값은 None
- `fetch_k` (int): 필터링 전에 가져올 문서 수. 기본값은 20
- `**kwargs`: 추가 키워드 인자

**반환값**

- `List[Document]`: 쿼리와 가장 유사한 문서 리스트

**동작 방식**

1. `similarity_search_with_score` 메서드를 내부적으로 호출하여 유사도 점수와 함께 문서를 검색합니다.
2. 검색 결과에서 점수를 제외하고 문서만 추출하여 반환합니다.

**주요 특징**

- `filter` 매개변수를 사용하여 메타데이터 기반의 필터링이 가능합니다.
- `fetch_k`를 통해 필터링 전 검색할 문서 수를 조절할 수 있어, 필터링 후 원하는 수의 문서를 확보할 수 있습니다.

**사용 시 고려사항**

- 검색 성능은 사용된 임베딩 모델의 품질에 크게 의존합니다.
- 대규모 데이터셋에서는 `k`와 `fetch_k` 값을 적절히 조정하여 검색 속도와 정확도의 균형을 맞추는 것이 중요합니다.
- 복잡한 필터링이 필요한 경우, `filter` 매개변수에 커스텀 함수를 전달하여 세밀한 제어가 가능합니다.

**최적화 팁**

- 자주 사용되는 쿼리에 대해서는 결과를 캐싱하여 반복적인 검색 속도를 향상시킬 수 있습니다.
- `fetch_k`를 너무 크게 설정하면 검색 속도가 느려질 수 있으므로, 적절한 값을 실험적으로 찾는 것이 좋습니다.


### 텍스트로부터 추가 (add_texts)

`add_texts` 메서드는 텍스트를 임베딩하고 벡터 저장소에 추가하는 기능을 제공합니다.

**매개변수**

- `texts` (Iterable[str]): 벡터 저장소에 추가할 텍스트 이터러블
- `metadatas` (Optional[List[dict]]): 텍스트와 연관된 메타데이터 리스트 (선택적)
- `ids` (Optional[List[str]]): 텍스트의 고유 식별자 리스트 (선택적)
- `**kwargs`: 추가 키워드 인자

**반환값**

- `List[str]`: 벡터 저장소에 추가된 텍스트의 ID 리스트

**동작 방식**

1. 입력받은 텍스트 이터러블을 리스트로 변환합니다.
2. `_embed_documents` 메서드를 사용하여 텍스트를 임베딩합니다.
3. `__add` 메서드를 호출하여 임베딩된 텍스트를 벡터 저장소에 추가합니다.

### 문서 삭제 (Delete Documents)

`delete` 메서드는 벡터 저장소에서 지정된 ID에 해당하는 문서를 삭제하는 기능을 제공합니다.

**매개변수**

- `ids` (Optional[List[str]]): 삭제할 문서의 ID 리스트
- `**kwargs`: 추가 키워드 인자 (이 메서드에서는 사용되지 않음)

**반환값**

- `Optional[bool]`: 삭제 성공 시 True, 실패 시 False, 구현되지 않은 경우 None

**동작 방식**

1. 입력된 ID의 유효성을 검사합니다.
2. 삭제할 ID에 해당하는 인덱스를 찾습니다.
3. FAISS 인덱스에서 해당 ID를 제거합니다.
4. 문서 저장소에서 해당 ID의 문서를 삭제합니다.
5. 인덱스와 ID 매핑을 업데이트합니다.

**주요 특징**

- ID 기반 삭제로 정확한 문서 관리가 가능합니다.
- FAISS 인덱스와 문서 저장소 양쪽에서 삭제를 수행합니다.
- 삭제 후 인덱스 재정렬을 통해 데이터 일관성을 유지합니다.

**주의사항**

- 삭제 작업은 되돌릴 수 없으므로 신중하게 수행해야 합니다.
- 동시성 제어가 구현되어 있지 않아 다중 스레드 환경에서 주의가 필요합니다.


## 저장 및 로드

### 로컬 저장 (Save Local)

`save_local` 메서드는 FAISS 인덱스, 문서 저장소, 그리고 인덱스-문서 ID 매핑을 로컬 디스크에 저장하는 기능을 제공합니다.

**매개변수**

- `folder_path` (str): 저장할 폴더 경로
- `index_name` (str): 저장할 인덱스 파일 이름 (기본값: "index")

**동작 방식**

1. 지정된 폴더 경로를 생성합니다 (이미 존재하는 경우 무시).
2. FAISS 인덱스를 별도의 파일로 저장합니다.
3. 문서 저장소와 인덱스-문서 ID 매핑을 pickle 형식으로 저장합니다.

**사용 시 고려사항**

- 저장 경로에 대한 쓰기 권한이 필요합니다.
- 대용량 데이터의 경우 저장 공간과 시간이 상당히 소요될 수 있습니다.
- pickle 사용으로 인한 보안 위험을 고려해야 합니다.

### 로컬에서 불러오기 (Load Local)

`load_local` 클래스 메서드는 로컬 디스크에 저장된 FAISS 인덱스, 문서 저장소, 그리고 인덱스-문서 ID 매핑을 불러오는 기능을 제공합니다.

**매개변수**

- `folder_path` (str): 불러올 파일들이 저장된 폴더 경로
- `embeddings` (Embeddings): 쿼리 생성에 사용할 임베딩 객체
- `index_name` (str): 불러올 인덱스 파일 이름 (기본값: "index")
- `allow_dangerous_deserialization` (bool): pickle 파일 역직렬화 허용 여부 (기본값: False)

**반환값**

- `FAISS`: 로드된 FAISS 객체

**동작 방식**

1. 역직렬화의 위험성을 확인하고 사용자의 명시적 허가를 요구합니다.
2. FAISS 인덱스를 별도로 불러옵니다.
3. pickle을 사용하여 문서 저장소와 인덱스-문서 ID 매핑을 불러옵니다.
4. 불러온 데이터로 FAISS 객체를 생성하여 반환합니다.


### FAISS 객체 병합 (Merge From)

`merge_from` 메서드는 현재 FAISS 객체에 다른 FAISS 객체를 병합하는 기능을 제공합니다.

**매개변수**

- `target` (FAISS): 현재 객체에 병합할 대상 FAISS 객체

**동작 방식**

1. 문서 저장소의 병합 가능 여부를 확인합니다.
2. 기존 인덱스의 길이를 기준으로 새로운 문서들의 인덱스를 설정합니다.
3. FAISS 인덱스를 병합합니다.
4. 대상 FAISS 객체의 문서와 ID 정보를 추출합니다.
5. 추출한 정보를 현재 문서 저장소와 인덱스-문서 ID 매핑에 추가합니다.

**주요 특징**

- 두 FAISS 객체의 인덱스, 문서 저장소, 인덱스-문서 ID 매핑을 모두 병합합니다.
- 인덱스 번호의 연속성을 유지하면서 병합합니다.
- 문서 저장소의 병합 가능 여부를 사전에 확인합니다.


**주의사항**

- 병합 대상 FAISS 객체와 현재 객체의 구조가 호환되어야 합니다.
- 중복 ID 처리에 주의해야 합니다. 현재 구현에서는 중복 검사를 하지 않습니다.
- 병합 과정에서 예외가 발생하면 부분적으로 병합된 상태가 될 수 있습니다.


`merge_from` 를 사용하여 2개의 db 를 병합합니다.


## 검색기로 변환 (as_retriever)

`as_retriever` 메서드는 현재 벡터 저장소를 기반으로 `VectorStoreRetriever` 객체를 생성하는 기능을 제공합니다.

**매개변수**

- `**kwargs`: 검색 함수에 전달할 키워드 인자
  - `search_type` (Optional[str]): 검색 유형 (`"similarity"`, `"mmr"`, `"similarity_score_threshold"`)
  - `search_kwargs` (Optional[Dict]): 검색 함수에 전달할 추가 키워드 인자

**반환값**

- `VectorStoreRetriever`: 벡터 저장소 기반의 검색기 객체

**주요 기능**

1. 다양한 검색 유형 지원:
   - `"similarity"`: 유사도 기반 검색 (기본값)
   - `"mmr"`: Maximal Marginal Relevance 검색
   - `"similarity_score_threshold"`: 임계값 기반 유사도 검색

2. 검색 매개변수 커스터마이징:
   - `k`: 반환할 문서 수
   - `score_threshold`: 유사도 점수 임계값
   - `fetch_k`: MMR 알고리즘에 전달할 문서 수
   - `lambda_mult`: MMR 다양성 조절 파라미터
   - `filter`: 문서 메타데이터 기반 필터링

**사용 시 고려사항**

- 검색 유형과 매개변수를 적절히 선택하여 검색 결과의 품질과 다양성을 조절할 수 있습니다.
- 대규모 데이터셋에서는 `fetch_k`와 `k` 값을 조절하여 성능과 정확도의 균형을 맞출 수 있습니다.
- 필터링 기능을 활용하여 특정 조건에 맞는 문서만 검색할 수 있습니다.

**최적화 팁**

- MMR 검색 시 `fetch_k`를 높이고 `lambda_mult`를 조절하여 다양성과 관련성의 균형을 맞출 수 있습니다.
- 임계값 기반 검색을 사용하여 높은 관련성을 가진 문서만 반환할 수 있습니다.

**주의사항**

- 부적절한 매개변수 설정은 검색 성능이나 결과의 품질에 영향을 줄 수 있습니다.
- 대규모 데이터셋에서 높은 `k` 값 설정은 검색 시간을 증가시킬 수 있습니다.