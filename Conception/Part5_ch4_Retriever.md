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
