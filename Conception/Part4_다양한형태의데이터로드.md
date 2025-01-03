# Part4 다양한 형태의 데이터 로드

## 01. 다큐먼트 로더의 종류, 기본 구조, Documnet 구조

- 프로젝트마다 알맞은 다큐멘트 로더를 찾아서 사용하자!

- 사람들이 자주 찾는 로더
    - https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
    - https://python.langchain.com/v0.1/docs/integrations/document_loaders/
- Deprecate -> 이제 안하는애들

- PyPDFLoader 여기서 BaseLoader -> 이부분을 잘 이해해야한다
    - 출처 페이지정보 어느 파일에서 가지고왔는지 까지 가지고온다

- docs[5].page_content
    - 내용확인

- docs[5].metadata 
    - 메타 데이터 확인    

- 모든 로더에서 가장 중요한 객체 Document


- 다양한 PDF로더중 어떤걸 선택해야하는지?
    - 하나씩해보면서 가장 정교한 pdf split된걸 확인
    - 메타데이터를 확인해서 필요한게 꼭 있는지 확인


### load()

- 문서를 로드하여 반환합니다.
- 반환된 결과는 `List[Document]` 형태입니다.

### load_and_split()

- splitter 를 사용하여 문서를 분할하고 반환합니다.
- 반환된 결과는 `List[Document]` 형태입니다.


### lazy_load()

- generator 방식으로 문서를 로드합니다.


### aload()

- 비동기(Async) 방식의 문서 로드


## 02.  Documentsm Document_loader의 구조 이해하기
- Document 객체
    - page_content : 문서에서 로드한 내용이 들어감 (key)
    - metadata : 다양한 속성들이 들어감
    - page_content와 metadata가 들어가있다고 생각하면 된다.

```
{'id': None,
 'metadata': {},
 'page_content': '안녕하세요? 이건 랭체인의 도큐먼드 입니다',
 'type': 'Document'}
```

- 메타데이터 넣기

```python

from langchain_core.documents import Document

document = Document(page_content="안녕하세요? 이건 랭체인의 도큐먼드 입니다", metadata={"source" : "Teddy_note"})

# 메타데이터 추가
document.metadata["source"] = "TeddyNote"
document.metadata["page"] = 1
document.metadata["author"] = "Teddy"
```

### Document Loader

다양한 파일의 형식으로부터 불러온 내용을 문서(Document) 객체로 변환하는 역할을 합니다.

#### 주요 Loader 

- PyPDFLoader: PDF 파일을 로드하는 로더입니다.
- CSVLoader: CSV 파일을 로드하는 로더입니다.
- UnstructuredHTMLLoader: HTML 파일을 로드하는 로더입니다.
- JSONLoader: JSON 파일을 로드하는 로더입니다.
- TextLoader: 텍스트 파일을 로드하는 로더입니다.
- DirectoryLoader: 디렉토리를 로드하는 로더입니다.

#### load()

- 문서를 로드하여 반환합니다.
- 반환된 결과는 `List[Document]` 형태입니다.
- list단위 -> 문서의 page 단위로 진행 예 20페이지면 -> 20개의 list원소가 생김

#### load_and_split()

- splitter 를 사용하여 문서를 분할하고 반환합니다.
- 반환된 결과는 `List[Document]` 형태입니다.
- RecursiveCharacterTextSplitter : 문서를 로드하면서 동시에 짤르는(chunk) 역할을 진행함

### load만 할껀지 load와 split을 같이 동시에 진행할껀지

#### lazy_load()

- 만약 100만 패이지라고 가정하면, 메모리 부하가 너무 심하다.
- lazy_load -> 올리고 지우고 / 이를 통해 메모리 부하 없애줌
- generator 방식으로 문서를 로드합니다.


#### aload()

- 비동기(Async) 방식의 문서 로드


## 03. PDF로더

- 점부다 base_loader를 받고 실행
#### AutoRAG 팀에서의 PDF 실험

- AutoRAG 에서 진행한 실험을 토대로 작성한 순위표

- 아래 표기된 숫자는 등수를 나타냅니다. (The lower, the better)

| | PDFMiner | PDFPlumber | PyPDFium2 | PyMuPDF | PyPDF2 |
|----------|:---------:|:----------:|:---------:|:-------:|:-----:|
| Medical  | 1         | 2          | 3         | 4       | 5     |
| Law      | 3         | 1          | 1         | 3       | 5     |
| Finance  | 1         | 2          | 2         | 4       | 5     |
| Public   | 1         | 1          | 1         | 4       | 5     |
| Sum      | 5         | 5          | 7         | 15      | 20    |

- 출처: [AutoRAG Medium 블로그](https://velog.io/@autorag/PDF-%ED%95%9C%EA%B8%80-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%B6%94%EC%B6%9C-%EC%8B%A4%ED%97%98#%EC%B4%9D%ED%8F%89)

## PyPDF

- 여기에서는 `pypdf`를 사용하여 PDF를 문서 배열로 로드하며, 각 문서는 `page` 번호와 함께 페이지 내용 및 메타데이터를 포함합니다.
- 일부 PDF에는 스캔된 문서나 그림 내에 텍스트 이미지가 포함되어 있습니다. `rapidocr-onnxruntime` 패키지를 사용하여 이미지에서 텍스트를 추출할 수도 있습니다.

- pdf에서 text가 인식이 안되는 경우는 extract_images를 통해 이미지 ocr로 뽑아온다.
```python
loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
```
- extract_image 결과를 텍스트로 가지고옴
```
LayoutParser : A Uniﬁed Toolkit for DL-Based DIA 5
Table 1: Current layout detection models in the LayoutParser model zoo
Dataset Base Model1Large Model Notes
PubLayNet [38] F / M M Layouts of modern scientiﬁc documents
PRImA [3] M - Layouts of scanned modern magazines and scientiﬁc reports
Newspape
```

#### PyMuPDF
- **PyMuPDF** 는 속도 최적화가 되어 있으며, PDF 및 해당 페이지에 대한 자세한 메타데이터를 포함하고 있습니다. 페이지 당 하나의 문서를 반환합니다:

- 몇개찍어보고 얼마나 잘 가져오는지 꼭 확인해야한다!!

#### Unstructured

- [Unstructured](https://unstructured-io.github.io/unstructured/)는 Markdown이나 PDF와 같은 비구조화된 또는 반구조화된 파일 형식을 다루기 위한 공통 인터페이스를 지원합니다. 

- LangChain의 [UnstructuredPDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html)는 Unstructured와 통합되어 PDF 문서를 LangChain [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html) 객체로 파싱합니다.

- 내부적으로 비정형에서는 텍스트 청크마다 서로 다른 "**요소**"를 만듭니다. 기본적으로 이들은 결합되어 있지만 `mode="elements"`를 지정하여 쉽게 분리할 수 있습니다.
```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# UnstructuredPDFLoader 인스턴스 생성
loader = UnstructuredPDFLoader(FILE_PATH)

# 데이터 로드
docs = loader.load()

# 문서의 내용 출력
print(docs[0].page_content[:300])

# UnstructuredPDFLoader 인스턴스 생성(mode="elements")
loader = UnstructuredPDFLoader(FILE_PATH, mode="elements")

# 데이터 로드
docs = loader.load()

# 문서의 내용 출력
print(docs[0].page_content)
```

- 항상 메타데이터에서 무엇을 줄 수 있는지 확인해보자!

#### PyPDFium2
- 잘안씀


#### PDFMiner

- 한글 데이터를 잘 가져오는 편

- 페이지정보가 metadata가 없음

- **PDFMiner**를 사용하여 HTML 텍스트 생성

- 이 방법은 출력된 HTML 콘텐츠를 `BeautifulSoup`을 통해 파싱함으로써 글꼴 크기, 페이지 번호, PDF 헤더/푸터 등에 대한 보다 구조화되고 풍부한 정보를 얻을 수 있게 하여 텍스트를 의미론적으로 섹션으로 분할하는 데 도움이 될 수 있습니다.

- html으로 가지고 와서 beaturifulsoup으로 처리가능함
- 내가 원하는 방식으로 커스텀이 가능함
- html자체를 커스텀하기 때문에 / 무엇이 중요한지 설명해 줄 수 있음

#### PyPDF 디렉토리

- 디렉토리에서 PDF를 로드하세요

#### PDFPlumber

- PyMuPDF와 마찬가지로, 출력 문서는 PDF와 그 페이지에 대한 자세한 메타데이터를 포함하며, 페이지 당 하나의 문서를 반환합니다.

- 바운딩 박스만 가지고와서 / 중요한 부분을 표시할 수 있다.
- 이미지 차트 처리하는거 8월 주주총회 참조s