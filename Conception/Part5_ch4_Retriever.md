# Retrieve
- retrieve pdf참조
- 가장 중요한 파트
- LLM이 검색된 기반의 문서로 답변
- 따라서 검색을 아주 잘해야 좋은 LLM 성능이 나옴
- 필요하지않은 정보를 제거해줘야함


## 검색기의 필요성
1. 정확한 정보 제공 
2. 응답 시간 단출
3. 최적화 -> K의 검색건수 -> K가 만약 30이라면, 문서의 5군데로 필요로한다면, 25개가 불필요하게됨 -> 할루시네이션으로 이루어짐
    - 따라서 필요한 K를 줄여가면서 최적의 정보를 찾아내는게 중요함


## 동작 방식
- 질문의 벡터화 : 사용자의 질문을 벡터 형태로 변환, 이 과정은 임베딩 단계과 유사한 기술을 사용하여 진행, 변환된 질문 벡터를 기준점으로 사용
- 백터의 유사성 비교 : 저장된 문서 벡터들과 질ㄹ문 벡터사이의 유사성을 계산한다. 이는 주로 cossim, Max Marginal Relevance 등의 수학적인 방법등이 사용


## Sparse vs Dense Retriever
    - 의미 vs 키워드
    - 키워드 검색의 비중을 크게할때도 있고, 아닐때도 있다.
    - 하이브리드 서치를 통해 섞어서 사용이 가능하다.

    - Sparse Retrieveer : TD-IDF,BM25 가많이 쓰임



# 02. 벡터스토어 기반 검색기(VectorStore-backed Retriever)

**VectorStore 지원 검색기** 는 vector store를 사용하여 문서를 검색하는 retriever입니다.

Vector store에 구현된 **유사도 검색(similarity search)** 이나 **MMR** 과 같은 검색 메서드를 사용하여 vector store 내의 텍스트를 쿼리합니다.



### VectorStore에서 VectorStoreRetriever 초기화(as_retriever)

`as_retriever` 메서드는 VectorStore 객체를 기반으로 VectorStoreRetriever를 초기화하고 반환합니다. 이 메서드를 통해 다양한 검색 옵션을 설정하여 사용자의 요구에 맞는 문서 검색을 수행할 수 있습니다.

**매개변수(parameters)**

- `**kwargs`: 검색 함수에 전달할 키워드 인자
  - `search_type`: 검색 유형 ("similarity(기본값)", "mmr", "similarity_score_threshold")
  - `search_kwargs`: 추가 검색 옵션
    - `k`: 반환할 문서 수 (기본값: 4)
    - `score_threshold`: similarity_score_threshold 검색의 최소 유사도 임계값, 어떤 문서를 갖고올지
    - `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20) , 기본으로 20개 문서 고르고 최종 k개
    - `lambda_mult`: MMR 결과의 다양성 조절 (0-1 사이, 기본값: 0.5) 0 다양성 없음, 1 다양성 많음
    - `filter`: 문서 메타데이터 기반 필터링, ex A,B,C문서에서 A문서만 쓰고싶을때

**반환값(return)**

- `VectorStoreRetriever`: 초기화된 VectorStoreRetriever 객체

**참고**

- 다양한 검색 전략 구현 가능 (유사도, MMR, 임계값 기반)
- MMR (Maximal Marginal Relevance) 알고리즘으로 검색 결과의 다양성 조절 가능
- 메타데이터 필터링으로 특정 조건의 문서만 검색 가능
- `tags` 매개변수를 통해 검색기에 태그 추가 가능

**주의사항**

- `search_type`과 `search_kwargs`의 적절한 조합 필요
- MMR 사용 시 `fetch_k`와 `k` 값의 균형 조절 필요
- `score_threshold` 설정 시 너무 높은 값은 검색 결과가 없을 수 있음
- 필터 사용 시 데이터셋의 메타데이터 구조 정확히 파악 필요
- `lambda_mult` 값이 0에 가까울수록 다양성이 높아지고, 1에 가까울수록 유사성이 높아짐

## 동적 설정(Configurable)

- invoke시점에서 동적으로 원하는 설정을 할 수 있음
- 검색 설정을 동적으로 조정하기 위해 `ConfigurableField` 를 사용합니다.
- `ConfigurableField` 는 검색 매개변수의 고유 식별자, 이름, 설명을 설정하는 역할을 합니다.
- 검색 설정을 조정하기 위해 `config` 매개변수를 사용하여 검색 설정을 지정합니다.
- 검색 설정은 `config` 매개변수에 전달된 딕셔너리의 `configurable` 키에 저장됩니다.
- 검색 설정은 검색 쿼리와 함께 전달되며, 검색 쿼리에 따라 동적으로 조정됩니다.

### Upstage 임베딩과 같이 Query & Passage embedding model 이 분리된 경우

기본 retriever는 쿼리와 문서에 대해 동일한 임베딩 모델을 사용합니다.

하지만 쿼리와 문서에 대해 서로 다른 임베딩 모델을 사용하는 경우가 있습니다.

이러한 경우에는 쿼리 임베딩 모델을 사용하여 쿼리를 임베딩하고, 문서 임베딩 모델을 사용하여 문서를 임베딩합니다.

이렇게 하면 쿼리와 문서에 대해 서로 다른 임베딩 모델을 사용할 수 있습니다.

## 03. 문맥 압축 검색기(ContextualCompressionRetriever)

- 20개 -> 너무많아 -> 5개 걸러야하는데 너무 어려움
- K는 넉넉하게 가고(10개)
- 실제로 우리가 필요한 5군데가 있다면
- 압축해서 추려가지고 넣어주자!
- LLM기반이 있고, embedding 기반이 있다
- LLM 기반시 토큰 비용이 있지만, 알짜배기 정보만 넣어줄 수 있다.
- embedding filter -> 유사도기반으로 한번더 해서, 쿼리와 유사성있는것을 가지고온다.

검색 시스템에서 직면하는 어려움 중 하나는 데이터를 시스템에 수집할 때 어떤 특정 질의를 처리해야 할지 미리 알 수 없다는 점입니다.

이는 질의와 가장 관련성이 높은 정보가 많은 양의 무관한 텍스트를 포함한 문서에 묻혀 있을 수 있음을 의미합니다.

이러한 전체 문서를 애플리케이션에 전달하면 더 비용이 많이 드는 LLM 호출과 품질이 낮은 응답으로 이어질 수 있습니다.

`ContextualCompressionRetriever` 은 이 문제를 해결하기 위해 고안되었습니다.

아이디어는 간단합니다. 검색된 문서를 그대로 즉시 반환하는 대신, 주어진 질의의 맥락을 사용하여 문서를 압축함으로써 관련 정보만 반환되도록 할 수 있습니다.

여기서 "압축"은 개별 문서의 내용을 압축하는 것과 문서를 전체적으로 필터링하는 것 모두를 의미합니다.

`ContextualCompressionRetriever` 는 질의를 base retriever에 전달하고, 초기 문서를 가져와 Document Compressor를 통과시킵니다.

Document Compressor는 문서 목록을 가져와 문서의 내용을 줄이거나 문서를 완전히 삭제하여 목록을 축소합니다.

> 출처: https://drive.google.com/uc?id=1CtNgWODXZudxAWSRiWgSGEoTNrUFT98v

![](../10-Retriever/images/01-Contextual-Compression.jpeg)


### 맥락적 압축(ContextualCompression)

`LLMChainExtractor` 를 활용하여 생성한 `DocumentCompressor` 를 retriever 에 적용한 것이 바로 `ContextualCompressionRetriever` 입니다.


### LLM 을 활용한 문서 필터링

### `LLMChainFilter`

`LLMChainFilter`는 초기에 검색된 문서 중 어떤 문서를 필터링하고 어떤 문서를 반환할지 결정하기 위해 LLM 체인을 사용하는 보다 단순하지만 강력한 압축기입니다.

이 필터는 문서 내용을 변경(압축)하지 않고 문서를 **선택적으로 반환** 합니다.

### `EmbeddingsFilter`

각각의 검색된 문서에 대해 추가적인 LLM 호출을 수행하는 것은 비용이 많이 들고 속도가 느립니다.

`EmbeddingsFilter`는 문서와 쿼리를 임베딩하고 쿼리와 충분히 유사한 임베딩을 가진 문서만 반환함으로써 더 저렴하고 빠른 옵션을 제공합니다.

이를 통해 검색 결과의 관련성을 유지하면서도 계산 비용과 시간을 절약할 수 있습니다.


`EmbeddingsFilter` 와 `ContextualCompressionRetriever` 를 사용하여 관련 문서를 압축하고 검색하는 과정입니다.

- `EmbeddingsFilter` 를 사용하여 지정된 **유사도 임계값(0.86)** 이상인 문서를 필터링 합니다.


## 파이프라인 생성(압축기+문서 변환기)

`DocumentCompressorPipeline` 을 사용하면 여러 compressor를 순차적으로 결합할 수 있습니다.

Compressor와 함께 `BaseDocumentTransformer`를 파이프라인에 추가할 수 있는데, 이는 맥락적 압축을 수행하지 않고 단순히 문서 집합에 대한 변환을 수행합니다.

예를 들어, `TextSplitter`는 문서를 더 작은 조각으로 분할하기 위해 document transformer로 사용될 수 있으며, `EmbeddingsRedundantFilter`는 문서 간의 임베딩 유사성(기본값: 0.95 유사도 이상을 중복 문서로 간주) 을 기반으로 중복 문서를 필터링하는 데 사용될 수 있습니다.

아래에서는 먼저 문서를 더 작은 청크로 분할한 다음, 중복 문서를 제거하고, 쿼리와의 관련성을 기준으로 필터링하여 compressor pipeline을 생성합니다.


`ContextualCompressionRetriever`를 초기화하며, `base_compressor`로 `pipeline_compressor`를, `base_retriever`로 `retriever`를 사용합니다.


# 04.앙상블 검색기(Ensemble Retriever)

`EnsembleRetriever`는 여러 검색기를 결합하여 더 강력한 검색 결과를 제공하는 LangChain의 기능입니다. 이 검색기는 다양한 검색 알고리즘의 장점을 활용하여 단일 알고리즘보다 더 나은 성능을 달성할 수 있습니다.

**주요 특징**
1. 여러 검색기 통합: 다양한 유형의 검색기를 입력으로 받아 결과를 결합합니다.
2. 결과 재순위화: [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 알고리즘을 사용하여 결과의 순위를 조정합니다.
3. 하이브리드 검색(hybrid search): 주로 `sparse retriever`(예: BM25)와 `dense retriever`(예: 임베딩 유사도)를 결합하여 사용합니다.

**장점**
- Sparse retriever: 키워드 기반 검색에 효과적
- Dense retriever: 의미적 유사성 기반 검색에 효과적

이러한 상호 보완적인 특성으로 인해 `EnsembleRetriever`는 다양한 검색 시나리오에서 향상된 성능을 제공할 수 있습니다.

자세한 내용은 [LangChain 공식 문서](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble)를 참조하세요.


- `ensemble_retriever` 객체의 `get_relevant_documents()` 메서드를 호출하여 관련성 높은 문서를 검색합니다.

## 런타임 Config 변경

런타임에서도 retriever 의 속성을 변경할 수 있습니다. 이는 `ConfigurableField` 클래스를 사용하여 가능합니다.

- `weights` 매개변수를 `ConfigurableField` 객체로 정의합니다.
  - 필드의 ID는 "ensemble_weights"로 설정합니다.

# 05. 긴 문맥 재정렬(LongContextReorder)

모델의 아키텍처와 상관없이, 10개 이상의 검색된 문서를 포함할 경우 성능이 상당히 저하됩니다.

간단히 말해, 모델이 긴 컨텍스트 중간에 있는 관련 정보에 접근해야 할 때, 제공된 문서를 무시하는 경향이 있습니다.

자세한 내용은 다음 논문을 참조하세요

- https://arxiv.org/abs/2307.03172

이 문제를 피하기 위해, 검색 후 문서의 순서를 재배열하여 성능 저하를 방지할 수 있습니다.

- `Chroma` 벡터 저장소를 사용하여 텍스트 데이터를 저장하고 검색할 수 있는 `retriever`를 생성합니다.
- `retriever`의 `invoke` 메서드를 사용하여 주어진 쿼리에 대해 관련성이 높은 문서를 검색합니다.

`LongContextReorder` 클래스의 인스턴스인 `reordering`을 생성합니다.

- `reordering.transform_documents(docs)`를 호출하여 문서 목록 `docs`를 재정렬합니다.
  - 덜 관련된 문서는 목록의 중간에 위치하고, 더 관련된 문서는 시작과 끝에 위치하도록 재정렬됩니다.

# 06. Parent Document Retriever

- 한개의 쿼리에 다양한 쿼리를 llm이 생성해서 새로운 쿼리를 만들어 정보를 찾는다.
**문서 검색과 문서 분할의 균형 잡기**

문서 검색 과정에서 문서를 적절한 크기의 조각(청크)으로 나누는 것은 다음의 **상충되는 두 가지 중요한 요소를 고려**해야 합니다.

1. 작은 문서를 원하는 경우: 이렇게 하면 문서의 임베딩이 그 의미를 가장 정확하게 반영할 수 있습니다. 문서가 너무 길면 임베딩이 의미를 잃어버릴 수 있습니다.
2. 각 청크의 맥락이 유지되도록 충분히 긴 문서를 원하는 경우입니다.

**`ParentDocumentRetriever`의 역할**

이 두 요구 사항 사이의 균형을 맞추기 위해 `ParentDocumentRetriever`라는 도구가 사용됩니다. 이 도구는 문서를 작은 조각으로 나누고, 이 조각들을 관리합니다. 검색을 진행할 때는, 먼저 이 작은 조각들을 찾아낸 다음, 이 조각들이 속한 원본 문서(또는 더 큰 조각)의 식별자(ID)를 통해 전체적인 맥락을 파악할 수 있습니다.

여기서 '부모 문서'란, 작은 조각이 나누어진 원본 문서를 말합니다. 이는 전체 문서일 수도 있고, 비교적 큰 다른 조각일 수도 있습니다. 이 방식을 통해 문서의 의미를 정확하게 파악하면서도, 전체적인 맥락을 유지할 수 있게 됩니다.

**정리**

- **문서 간의 계층 구조 활용**: `ParentDocumentRetriever`는 문서 검색의 효율성을 높이기 위해 문서 간의 계층 구조를 활용합니다.
- **검색 성능 향상**: 관련성 높은 문서를 빠르게 찾아내며, 주어진 질문에 대한 가장 적합한 답변을 제공하는 문서를 효과적으로 찾아낼 수 있습니다.
  문서를 검색할 때 자주 발생하는 두 가지 상충되는 요구 사항이 있습니다:

# 07. MultiQueryRetriever

거리 기반 벡터 데이터베이스 검색은 고차원 공간에서의 쿼리 임베딩(표현)과 '거리'를 기준으로 유사한 임베딩을 가진 문서를 찾는 방식입니다. 하지만 쿼리의 **세부적인 차이나 임베딩이 데이터의 의미를 제대로 포착하지 못할 경우, 검색 결과가 달라질 수** 있습니다. 또한, 이를 수동으로 조정하는 프롬프트 엔지니어링이나 튜닝 작업은 번거로울 수 있습니다.

이런 문제를 해결하기 위해, `MultiQueryRetriever` 는 주어진 사용자 입력 쿼리에 대해 다양한 관점에서 여러 쿼리를 자동으로 생성하는 LLM(Language Learning Model)을 활용해 프롬프트 튜닝 과정을 자동화합니다.

이 방식은 각각의 쿼리에 대해 관련 문서 집합을 검색하고, 모든 쿼리를 아우르는 고유한 문서들의 합집합을 추출해, 잠재적으로 관련된 더 큰 문서 집합을 얻을 수 있게 해줍니다. 

여러 관점에서 동일한 질문을 생성함으로써, `MultiQueryRetriever` 는 거리 기반 검색의 제한을 일정 부분 극복하고, 더욱 풍부한 검색 결과를 제공할 수 있습니다.

## 사용방법

`MultiQueryRetriever` 에 사용할 LLM을 지정하고 질의 생성에 사용하면, retriever가 나머지 작업을 처리합니다.


아래는 다중 쿼리를 생성하는 중간 과정을 디버깅하기 위하여 실행하는 코드입니다.

먼저 `"langchain.retrievers.multi_query"` 로거를 가져옵니다. 

이는 `logging.getLogger()` 함수를 사용하여 수행됩니다. 그 다음, 이 로거의 로그 레벨을 `INFO`로 설정하여, `INFO` 레벨 이상의 로그 메시지만 출력되도록 할 수 있습니다. 

이 코드는 `retriever_from_llm` 객체의 `invoke` 메서드를 사용하여 주어진 `question`과 관련된 문서를 검색합니다. 

검색된 문서들은 `unique_docs`라는 변수에 저장되며, 이 변수의 길이를 확인함으로써 검색된 관련 문서의 총 개수를 알 수 있습니다. 이 과정을 통해 사용자의 질문에 대한 관련 정보를 효과적으로 찾아내고 그 양을 파악할 수 있습니다.

장점
1. 유자가 쿼리를 친절하게 대답을 안해서, 후처리적으로 친절하게 진행
2. 검색할때, 날것의 질문보다는, llm이 만저진 질문을 통해 좀더 좋은 쿼리를 넣어줄 수 있다.


# 08.  다중 벡터저장소 검색기(MultiVectorRetriever)

- 정말 많이 사용함
- vector store + inmemory store 의 2가지 모두 운영
- 청킹된 문서들을 통해 검색을하면 여러개중 한개가 걸려 검색 -> RAG
- 그런데 다른 부분에 정보가 분산되어 있다면, 전체 페이지격의 문서를 반환
- parantdocumentretriver와 비슷함
- vectore store -> 문서를 아디화해서 저장, 문서전체의 페이지 : doc sotre 에저장
- abc라는 키워드를 찾을때, 작은 청크들을 가지고와서 쓴다.
- 둘다 사용가능
- 1,2,3 청크 -> A 페이지 -> 1,2,3을 요약
  - 요약본들은 embdeded vectore에 저장
  - 요약본 1,2,3 을 저장 -> 요약본에서 원본 문서를 찾는다
- 다양하게 가능

LangChain에서는 문서를 다양한 상황에서 효율적으로 쿼리할 수 있는 특별한 기능, 바로 `MultiVectorRetriever`를 제공합니다. 이 기능을 사용하면 문서를 여러 벡터로 저장하고 관리할 수 있어, 정보 검색의 정확도와 효율성을 대폭 향상시킬 수 있습니다. 

`MultiVectorRetriever`를 활용해 문서당 여러 벡터를 생성하는 몇 가지 방법을 살펴보겠습니다.

**문서당 여러 벡터 생성 방법 소개**

1. **작은 청크 생성**: 문서를 더 작은 단위로 나눈 후, 각 청크에 대해 별도의 임베딩을 생성합니다. 이 방식을 사용하면 문서의 특정 부분에 좀 더 세심한 주의를 기울일 수 있습니다. 이 과정은 `ParentDocumentRetriever`를 통해 구현할 수 있어, 세부 정보에 대한 탐색이 용이해집니다.

2. **요약 임베딩**: 각 문서의 요약을 생성하고, 이 요약으로부터 임베딩을 만듭니다. 이 요약 임베딩은 문서의 핵심 내용을 신속하게 파악하는 데 큰 도움이 됩니다. 문서 전체를 분석하는 대신 핵심적인 요약 부분만을 활용하여 효율성을 극대화할 수 있습니다.

3. **가설 질문 활용**: 각 문서에 대해 적합한 가설 질문을 만들고, 이 질문에 기반한 임베딩을 생성합니다. 특정 주제나 내용에 대해 깊이 있는 탐색을 원할 때 이 방법이 유용합니다. 가설 질문은 문서의 내용을 다양한 관점에서 접근하게 해주며, 더 광범위한 이해를 가능하게 합니다.

4. **수동 추가 방식**: 사용자가 문서 검색 시 고려해야 할 특정 질문이나 쿼리를 직접 추가할 수 있습니다. 이 방법을 통해 사용자는 검색 과정에서 보다 세밀한 제어를 할 수 있으며, 자신의 요구 사항에 맞춘 맞춤형 검색이 가능해집니다.

## Chunk + 원본 문서 검색

대용량 정보를 검색하는 경우, 더 작은 단위로 정보를 임베딩하는 것이 유용할 수 있습니다.

`MultiVectorRetriever` 를 통해 문서를 여러 벡터로 저장하고 관리할 수 있습니다.

`docstore` 에 원본 문서를 저장하고, `vectorstore` 에 임베딩된 문서를 저장합니다. 

이로써 문서를 더 작은 단위로 나누어 더 정확한 검색이 가능해집니다. 때에 따라서는 원본 문서의 내용을 조회할 수 있습니다.



## 가설 쿼리(Hypothetical Queries) 를 활용하여 문서 내용 탐색

LLM은 특정 문서에 대해 가정할 수 있는 질문 목록을 생성하는 데에도 사용될 수 있습니다.

이렇게 생성된 질문들은 임베딩(embedding)될 수 있으며, 이를 통해 문서의 내용을 더욱 깊이 있게 탐색하고 이해할 수 있습니다.

가정 질문 생성은 문서의 주요 주제와 개념을 파악하는 데 도움이 되며, 독자들이 문서 내용에 대해 더 많은 궁금증을 갖도록 유도할 수 있습니다.

# Self-querying

`SelfQueryRetriever` 는 자체적으로 질문을 생성하고 해결할 수 있는 기능을 갖춘 검색 도구입니다. 

이는 사용자가 제공한 자연어 질의를 바탕으로, `query-constructing` LLM chain을 사용해 구조화된 질의를 만듭니다. 그 후, 이 구조화된 질의를 기본 벡터 데이터 저장소(VectorStore)에 적용하여 검색을 수행합니다.

이 과정을 통해, `SelfQueryRetriever` 는 단순히 사용자의 입력 질의를 저장된 문서의 내용과 의미적으로 비교하는 것을 넘어서, 사용자의 질의에서 문서의 메타데이터에 대한 **필터를 추출** 하고, 이 필터를 실행하여 관련된 문서를 찾을 수 있습니다. 

[참고]

- LangChain 이 지원하는 셀프 쿼리 검색기(Self-query Retriever) 목록은 [여기](https://python.langchain.com/docs/integrations/retrievers/self_query) 에서 확인해 주시기 바랍니다.

- 연도, 상품종류 등등 쿼리에서 filtering이 필요할때 


# TimeWeightedVectorStoreRetriever

`TimeWeightedVectorStoreRetriever` 는 의미론적 유사성과 시간에 따른 감쇠를 결합해 사용하는 검색 도구입니다. 이를 통해 문서 또는 데이터의 **"신선함"** 과 **"관련성"** 을 모두 고려하여 결과를 제공합니다.

스코어링 알고리즘은 다음과 같이 구성됩니다

$\text{semantic\_similarity} + (1.0 - \text{decay\_rate})^{hours\_passed}$

여기서 `semantic_similarity` 는 문서 또는 데이터 간의 의미적 유사도를 나타내고, `decay_rate` 는 시간이 지남에 따라 점수가 얼마나 감소하는지를 나타내는 비율입니다. `hours_passed` 는 객체가 마지막으로 접근된 후부터 현재까지 경과한 시간(시간 단위)을 의미합니다.

이 방식의 주요 특징은, 객체가 마지막으로 접근된 시간을 기준으로 하여 **"정보의 신선함"** 을 평가한다는 점입니다. 즉, **자주 접근되는 객체는 시간이 지나도 높은 점수**를 유지하며, 이를 통해 **자주 사용되거나 중요하게 여겨지는 정보가 검색 결과 상위에 위치할 가능성이 높아집니다.** 이런 방식은 최신성과 관련성을 모두 고려하는 동적인 검색 결과를 제공합니다.

특히, `decay_rate` 는 리트리버의 객체가 생성된 이후가 아니라 **마지막으로 액세스된 이후 경과된 시간** 을 의미합니다. 즉, 자주 액세스하는 객체는 '최신'으로 유지됩니다.


## 낮은 감쇠율(low decay_rate)

- `decay rate` 가 낮다는 것은 (여기서는 극단적으로 0에 가깝게 설정할 것입니다) **기억이 더 오래 "기억될"** 것임을 의미합니다.

- `decay rate` 가 **0 이라는 것은 기억이 절대 잊혀지지 않는다**는 것을 의미하며, 이는 이 retriever를 vector lookup과 동등하게 만듭니다.

`TimeWeightedVectorStoreRetriever`를 초기화하며, 벡터 저장소, 감쇠율(`decay_rate`)을 매우 작은 값으로 설정하고, 검색할 벡터의 개수(k)를 1로 지정합니다.
