# RAG 프로세스 이해하기

## 1. RAG 프로세스 이해하기

- RAG를 왜써야하냐?
    - 할루시네이션 방지
    - 똑같은 질문이 들어 왔을때 정보를 참조해서 답변을 제공
        - 프롬프트가 **참고자료/문맥(context)**을 검색해서 답변을 한다.

- 1.1 그럼 어떻게 가져올까?
    - 전체 PDF를 넣을경우 엄청난양의 프롬프트양으로인해 과금이 올라간다.
    - 관련성 없는 데이터까지 넣어줄경우 중간에 길을 잃어버린다 lost in the middle
    - 따라서, 관련성이 높은 부분만 찾아서 넣어준다.
    - 특정 단락에 맞춰서 정보를 제공한다.
        - Chunk : 각 단위에 맞춰서 분할한다.
        - 유사도검색 -> Rank로 몇개 뽑아(k개지정) 
- 1.2 임배딩(임배딩.pdf)참조
    - 임베딩 하는 과정에서도 돈이 든다.
    - embedding 한 이후 VectorStore를 통해 저장해 둔다
    - 추 후에는 DB의 Store에 Query요청을 하면 된다.
- 1.3 텍스트 분할(Text Split)
    - Chunk overlap


## 2. Retrival Augmented Generation(RAG)
- PDF 참조
- VectorStore까지 저장

- 런타임단계(Runtime 단계)
    - DB는 이미 준비가 되어있는상태( Vector Store까지 저장해 놓은 단계)

- RETRIEVE : 넣어준 DB에서 필요한 정보들을 뽑아내는 단계
- 그 이후 Prompt를 통해 답을 제공

- 검색기의 필요성
    - 검색기.pdf
    - 동작방식
    - 1. 질문의 벡터화 -> 2. 벡터 유사성 비교 -> 3. 상위 문서 선정 -> 4. 문서정보
    - 정보검색 시스템 -> Sparse Retriever vs Dense Retriever
    - Dense Retriever 의미적인 부분 / Sparse Retriever Keyword로 탐색

- 프롬프트(Prompt) pdf
- LLM pdf
- Chain pdf

## 3. PDF 문서기반 QA RAG

## 실습