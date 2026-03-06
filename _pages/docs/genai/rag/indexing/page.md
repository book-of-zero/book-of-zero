---
layout: post
title: "RAG Indexing: chunking, embedding, and storage"
nav_order: 5
---

A 50-page PDF goes into your RAG system. The model hallucinates, misses key details, or mixes facts from unrelated sections. The root cause: bad chunking. How you split documents, encode them, and store them determines what the retrieval stage can find — and what the LLM can generate.

Chunking is not preprocessing. It's architecture. A chunk is your retrieval unit, your context boundary, and your quality ceiling. Get it wrong and no amount of prompt engineering or model scaling will save you.

---

## On this page

- [Key concepts](#key-concepts)
- [Data preprocessing & cleaning](#data-preprocessing--cleaning)
  - [Cleaning PDF documents](#cleaning-pdf-documents)
  - [Cleaning web scrapes](#cleaning-web-scrapes)
  - [Cleaning OCR output](#cleaning-ocr-output)
  - [Preprocessing checklist](#preprocessing-checklist)
- [Chunking strategies](#chunking-strategies)
  - [Character splitting](#character-splitting)
  - [Recursive character splitting](#recursive-character-splitting)
  - [Document-specific splitting](#document-specific-splitting)
  - [Semantic chunking](#semantic-chunking)
  - [Agentic splitting](#agentic-splitting)
  - [Alternative representations](#alternative-representations)
- [Embedding models](#embedding-models)
- [Metadata strategy](#metadata-strategy)
  - [What metadata to track](#what-metadata-to-track)
  - [Metadata structure](#metadata-structure)
  - [Contextual chunk headers (CCH)](#contextual-chunk-headers-cch)
- [Vector storage](#vector-storage)
- [Chunking evaluation](#chunking-evaluation)
- [Workflow: building an index](#workflow-building-an-index)
  - [Prototype pipeline](#prototype-pipeline)
  - [Production pipeline](#production-pipeline)
  - [Incremental updates](#incremental-updates)
- [Common pitfalls](#common-pitfalls)
- [Operations & planning](#operations--planning)
- [Best practices](#best-practices)

---

## Key concepts

**Chunk**: The atomic retrieval unit in RAG. A contiguous text segment that must be meaningful in isolation.

**Chunk size**: Target length in tokens or characters. Typical range: 256-800 tokens. Smaller chunks increase precision at the cost of context.

**Chunk overlap**: Tokens shared between consecutive chunks to prevent information loss at boundaries. Typical: 10-20% of chunk size.

**Embedding**: A dense vector encoding semantic meaning. Similar meanings cluster together in high-dimensional space.

**Vector database**: Storage optimized for approximate nearest neighbor (ANN) search over embeddings. Linear search doesn't scale past 10K chunks.

**Semantic chunking**: Splitting based on meaning via embedding similarity rather than syntax. 10-100x slower than recursive splitting but higher quality for heterogeneous documents.

---

## Data preprocessing & cleaning

Real-world documents are messy. PDFs often include headers, footers, and watermarks. OCR can introduce garbled text. Web scrapes often include nav menus and ads. Remove recurring boilerplate before chunking so embeddings focus on the content rather than repeated wrapper text. If every page says "CONFIDENTIAL", that term becomes much less useful for retrieval.

### Cleaning PDF documents

PDFs are common offenders: text extraction tools often preserve layout artifacts that should not be embedded.

#### Common PDF artifacts to remove

**Headers and footers**: page numbers ("Page 1", "1 of 87"), document titles, watermarks ("DRAFT", "CONFIDENTIAL"), dates.

**Layout artifacts**: horizontal separators, table borders, multi-column mixing.

**OCR errors** (scanned PDFs): character substitution, missing/extra spaces, phantom characters from image noise.

#### Cleaning pipeline

```python
import re
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser

def clean_pdf_text(pdf_path: str) -> str:
    config_parser = ConfigParser({"output_format": "markdown"})
    converter = PdfConverter(config=config_parser.generate_config_dict())
    rendered = converter(pdf_path)
    text = rendered.markdown

    text = re.sub(r'\b[Pp]age \d+\b', '', text)
    text = re.sub(r'\b\d+ of \d+\b', '', text)
    text = re.sub(r'\bCONFIDENTIAL\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyright © \d{4}.*', '', text)
    text = re.sub(r'[-─=]{3,}', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

<details class="boz-resource">
<summary>Advanced PDF preprocessing with ML-based parsing</summary>

For production systems with heterogeneous PDFs, use ML-based document parsing:

- **Docling** (IBM): ML-based document understanding, handles complex layouts
- **Azure Document Intelligence**: Cloud API, best for forms/tables/receipts
- **Google Document AI**: Cloud API, strong on tables/forms, supports 200+ languages
- **AWS Textract**: Similar to Azure, good for scanned documents

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(endpoint="<endpoint>", credential=AzureKeyCredential("<key>"))

with open("financial_report.pdf", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", body=f, content_type="application/pdf")

result = poller.result()
markdown = result.content
```

</details>

### Cleaning web scrapes

Web pages have navigation, ads, and boilerplate that pollute chunks. Remove nav elements, cookie banners, footers, and ads before chunking.

```python
from trafilatura import fetch_url, extract

def clean_web_page(url: str) -> str:
    downloaded = fetch_url(url)
    return extract(downloaded, include_comments=False, include_tables=True)
```

<details class="boz-resource">
<summary>Manual BeautifulSoup cleaning pipeline</summary>

```python
from bs4 import BeautifulSoup
import requests, re

def clean_web_page_manual(url: str) -> str:
    soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
        element.decompose()
    for element in soup.find_all(class_=re.compile(r'ad|banner|cookie|popup|subscribe', re.I)):
        element.decompose()
    main = soup.find('main') or soup.find('article') or soup.find('body')
    text = main.get_text(separator='\n', strip=True)
    return '\n\n'.join(line for line in text.split('\n') if line.strip())
```

</details>

### Cleaning OCR output

OCR quality varies widely. Common errors include character substitution (l/1/I, O/0, rn/m), missing spaces, and extra spaces.

These heuristics are corpus-specific examples, not universal defaults. Validate them on sample documents before applying them broadly.

```python
import re

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)         # collapse multiple spaces
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)    # rejoin hyphenated line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)           # collapse blank lines
    return text.strip()
```

Regex-based character fixes (l/1/I, O/0, rn/m) are tempting but fragile — they tend to break valid words. If OCR quality is poor, prefer re-scanning at higher resolution (300+ DPI), preprocessing images (deskew, denoise, binarize), or using stronger OCR systems such as Azure Computer Vision, AWS Textract, or surya.

### Preprocessing checklist

**Quality checks**: Sample documents from each format and source type, manually review cleaned text. Verify important content is not removed (tables, equations, technical terms). Generate test queries from sample chunks and confirm cleaned content can still answer them.

**Deduplication**: Remove exact and near-duplicate content before chunking. Repeated text (boilerplate paragraphs, copied sections across documents) inflates index size and skews retrieval toward redundant results. Use MinHash or exact-match hashing at the document or paragraph level.

**Automated validation**: Spot-check that cleaned text retains the bulk of original length — if cleaning removes a large fraction, investigate what is being stripped. Most frequent terms in cleaned output should be domain-relevant, not artifacts like "Page" or "Confidential". Embed sample chunks and verify similar chunks cluster together.

**Monitoring in production**: Track cleaning metrics (% artifacts removed, text length before/after). Set alert thresholds based on your corpus baseline — flag documents where text retention drops well below the norm.

**Over-cleaning vs under-cleaning**: Over-cleaning removes important content (dates, dollar amounts). Under-cleaning wastes embedding capacity on boilerplate. Heuristic-based cleaning (regex) is fast but brittle; ML-based parsing (marker, docling) is slower but handles edge cases. Invest in ML-based parsing for messy documents (scanned PDFs, web scrapes, OCR).

---

## Chunking strategies

Start with recursive character splitting for most use cases. Move to document-specific or semantic approaches when you've measured that chunking is the bottleneck.

### Character splitting

Split text every N characters. Baseline for ablation studies only — never use in production.

```python
def character_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks
```

Splits mid-sentence, mid-word, mid-thought. You get `"...the capital of France is Par"` and `"ris. The Eiffel Tower..."`. The retrieval system can't match either chunk to a query about France's capital.

### Recursive character splitting

Split on paragraph breaks, then sentences, then words, respecting a maximum chunk size. This is your default.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(document_text)
```

Tries paragraph boundaries first, falls back to newlines, then spaces, then characters. Chunks respect natural boundaries most of the time. The overlap ensures context spanning boundaries is captured. Using `from_tiktoken_encoder` measures chunk size in **tokens**, not characters — critical because a 1000-character chunk might be 250 or 400 tokens depending on content.

**Parameter tuning**: 256 tokens for high precision (FAQ, product specs), 500 tokens general use, 1000+ for long-context models. Overlap 10-20% of chunk size.

### Document-specific splitting

Use document structure (markdown headers, code syntax, PDF layout) to guide splitting when explicit hierarchy exists.

#### Markdown documents

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
md_chunks = markdown_splitter.split_text(markdown_document)
# Each chunk has metadata: {"h1": "Introduction", "h2": "Architecture"}
```

Headers signal topic boundaries. Metadata enables filtered search: "find authentication methods in the API section."

#### Code files

```python
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", chunk_size=1000, chunk_overlap=200,
)
code_chunks = code_splitter.split_text(python_code)
```

Prioritizes splitting at class definitions, then function definitions, then blank lines. Splitting mid-function produces unreadable chunks.

#### PDFs and multi-modal documents

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Preserves tables, sections, and layout hierarchy
markdown = result.document.export_to_markdown()
# Split by sections using MarkdownHeaderTextSplitter
```

Keeps tables intact, preserves section hierarchy. Use for scientific papers, reports, presentations.

### Semantic chunking

Split when the topic changes using embedding similarity, not at fixed intervals.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=25,
)
semantic_chunks = semantic_splitter.split_text(document_text)
```

The algorithm splits text into sentences, then combines each sentence with its neighbors (a buffer window of 1-2 sentences on each side) before embedding. This ensures each embedding captures local context, not just an isolated sentence. It then computes cosine similarities between consecutive combined-sentence embeddings and splits where similarity drops below the threshold (a topic boundary). Works well for heterogeneous documents where structure doesn't match semantics (transcripts, web scrapes, unformatted reports).

**Cost**: 10-100x slower than recursive splitting. For 10K words (100 sentences), that's 100 embedding calls. Run an ablation: if semantic chunking improves Hit Rate@5 by <5% over recursive splitting, it's not worth it.

### Agentic splitting

Use an LLM to decide chunk boundaries. Experimental and expensive ($0.01-0.10 per document, 5-10s latency, non-deterministic).

{% raw %}
```python
prompt = PromptTemplate.from_template("""
Split this text into coherent chunks of 300-800 tokens at topic boundaries.
Each chunk must be self-contained. Output JSON: [{{"chunk_text": "...", "summary": "..."}}]

Document: {document}
""")
response = llm.invoke(prompt.format(document=document_text))
```
{% endraw %}

Reserve for high-stakes applications (legal, medical) where chunking quality is critical and cost/latency don't matter. **Practical alternative**: use agentic chunking to label 100 documents, then train a lightweight boundary-detection model.

### Alternative representations

Instead of embedding raw chunks, create derivative representations that improve retrieval. This is especially useful for content that embeds poorly in its original form.

**Table summaries**: Tables embed poorly as raw text. Summarize them with an LLM, embed the summary, and store the original table for the generation step.

```python
summary = llm.invoke(f"Summarize this table in 2-3 sentences:\n{table_html}")
# Embed and index the summary
# Store the original table in metadata for the LLM to use at generation time
```

**Image descriptions**: Extract images from PDFs and generate text descriptions with a vision model. Embed the description for retrieval, pass the original image or description to the LLM.

**Document summaries**: Generate a summary of each full document and index it as a separate chunk. Queries about "the main argument" or "overall conclusion" match the summary instead of a random paragraph.

**Parent-child chunks**: Embed small chunks (200-400 tokens) for precise retrieval, but store a reference to their parent section (1000-2000 tokens). At retrieval time, return the parent instead of the child so the LLM has full context. Store the parent ID in chunk metadata.

```python
parent_chunks = splitter.split_text(document_text)  # 1500 tokens each
for i, parent in enumerate(parent_chunks):
    children = child_splitter.split_text(parent)  # 300 tokens each
    for child in children:
        index_chunk(child, metadata={"parent_id": i, "parent_text": parent})
# At retrieval: match on child embedding, return parent_text to LLM
```

The principle: your retrieval unit doesn't have to be a literal slice of the source document. If a derivative representation retrieves better, use it.

---

## Embedding models

### Model selection

Choose based on these criteria:

- **Benchmark scores**: Check the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for your task type (retrieval, STS, classification). A 2-point MTEB difference typically translates to 5-10% improvement in Hit Rate@5.
- **Dimensions**: Higher dimensions (1024-3072) capture more nuance but cost more storage and slower search. Many modern models support **Matryoshka representations**, letting you truncate embeddings to fewer dimensions with graceful quality degradation — useful for trading off storage against quality without retraining.
- **Hosting model**: API-based (OpenAI, Cohere, Voyage) for fast iteration. Self-hosted open-source (sentence-transformers, HuggingFace) for cost control and data privacy.
- **Language support**: Multilingual models exist but typically underperform monolingual ones. Match the model to your corpus language(s).
- **Instruction-tuned models**: Some models accept a task prefix (e.g., "Represent this document for retrieval:") that improves quality for specific use cases. Use these when your retrieval task is well-defined.

### Fine-tuning embeddings

Fine-tune when your domain diverges from training data (medical jargon, legal language, internal terminology) and you have 1000+ labeled query-document pairs.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('bge-large-en-v1.5')
train_examples = [
    InputExample(texts=["what is HIPAA?", "HIPAA is the Health Insurance Portability...", "The capital of France..."]),
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(train_objectives=[(train_dataloader, losses.TripletLoss(model))], epochs=3)
```

Skip if pre-trained models already hit your quality target or your domain is covered by general web text.

### Trade-offs

- **Dimensions**: 768 dims = 3KB/chunk (float32). For 1M chunks: 3GB. 3072 dims = 12GB. Higher dimensions capture more nuance but cost more storage and slower search. If available, use Matryoshka truncation to find the sweet spot.
- **Speed**: Batch embedding (10-100 chunks per call) amortizes API overhead. For large corpora, estimate total embedding time before committing to a model.
- **Quality**: Better embeddings improve retrieval, but if your LLM is bad at using context, fix the prompt first.

---

## Metadata strategy

Metadata transforms retrieval from "find similar text" to "find the right text from the right source at the right time." It enables filtering, ranking, attribution, and access control.

### What metadata to track

**Document-level** (applies to all chunks from a document):

- **source**: Document origin (filename, URL, database ID). Enables tracing chunks back to source.
- **doc_type**: Category (pdf, markdown, code, email). Enables type-specific filtering.
- **created_at / updated_at**: Timestamps (ISO 8601). Enables recency filtering.
- **author / owner**: Creator or responsible party. Enables authority-based ranking.
- **category / tags**: Topic classification. Enables domain filtering.
- **version**: Document version number. Enables version-specific retrieval.
- **access_level**: Permission level (public, internal, confidential). Enables basic access control.
- **roles / departments / permissions / owner_id**: RBAC fields for hard security filtering. See [retrieval RBAC]({{ site.baseurl }}/docs/genai/rag/retrieval/page/#hard-security-filtering-rbac) for implementation.

**Chunk-level** (specific to individual chunks):

- **chunk_index**: Position within document. Enables ordering and context reconstruction.
- **section / heading**: Section hierarchy. Enables section filtering.
- **page_number**: Page in original document (PDFs). Enables citation.
- **word_count**: Chunk size (word-level approximation via `len(chunk.split())`). Enables quality monitoring.

**Derived** (computed during indexing):

- **embedding_model**: Model used. Enables compatibility checking.
- **indexed_at**: When chunk was indexed. Enables incremental update tracking.
- **doc_hash**: SHA256 of source document. Enables change detection and idempotency.

### Metadata structure

Store metadata as flat key-value pairs, not nested objects. Vector databases index flat fields efficiently but struggle with nested structures.

```python
# Good: Flat metadata
metadata = {
    "source": "handbook.pdf",
    "doc_type": "pdf",
    "category": "hr",
    "created_at": "2024-01-15T10:30:00Z",
    "section": "Benefits",
}
```

<details class="boz-resource">
<summary>Full metadata attachment example</summary>

```python
chunks = splitter.split_text(document_text)
embeddings = model.encode(chunks)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    metadata = {
        "source": "hipaa_guide.pdf",
        "doc_type": "pdf",
        "category": "compliance",
        "author": "legal-team",
        "created_at": "2024-01-15T10:30:00Z",
        "version": "v2.3",
        "access_level": "internal",
        "roles": ["legal", "compliance", "engineering"],
        "departments": ["legal"],
        "permissions": ["read"],
        "owner_id": "legal-team",
        "chunk_index": i,
        "section": extract_section(chunk),
        "page_number": extract_page(chunk),
        "word_count": len(chunk.split()),
        "embedding_model": "all-mpnet-base-v2",
        "indexed_at": datetime.utcnow().isoformat(),
        "doc_hash": hashlib.sha256(document_text.encode()).hexdigest(),
    }

    client.upsert(
        collection_name="documents",
        points=[PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk, **metadata})]
    )
```

</details>

**Metadata enables**: filtered retrieval ("only legal-team PDFs"), recency boosting, attribution ("Source: hipaa_guide.pdf, Page 5"), access control via RBAC fields, deduplication via doc_hash.

### Contextual chunk headers (CCH)

Chunks often lack context about where they came from. "Nike has committed to reducing emissions by 50%" is meaningless without knowing it's from the "Nike Climate Impact Report 2025" under "Environmental Commitments." CCH prepends document and section context to chunks before embedding, significantly improving retrieval precision for ambiguous chunks. Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) showed ~49% reduction in retrieval failures using this approach.

**How it works**: Parse document structure, prepend context header (title + section hierarchy) to each chunk before embedding. Store both contextualized text (for embedding/search) and original text (for LLM generation).

```python
def create_contextual_chunks(document: str, doc_title: str, chunk_size: int = 500):
    sections = extract_document_structure(document, doc_title)
    all_chunks = []

    for section_header, section_text in sections:
        chunks = split_text(section_text, chunk_size=chunk_size, overlap=50)
        for i, chunk in enumerate(chunks):
            context_header = f"Document: {doc_title}\nSection: {section_header}\n\n"
            all_chunks.append({
                'contextualized': context_header + chunk,  # Embed this
                'original': chunk,                          # Return this to LLM
                'metadata': {'doc_title': doc_title, 'section': section_header, 'chunk_index': i}
            })
    return all_chunks
```

<details class="boz-resource">
<summary>Context header formats and full implementation</summary>

**Minimal** (title only):
```
Document: Nike Climate Impact Report 2025

Nike has committed to reducing emissions by 50% by 2030...
```

**Standard** (title + section):
```
Document: Nike Climate Impact Report 2025
Section: Environmental Commitments

Nike has committed to reducing emissions by 50% by 2030...
```

**Rich** (title + section + metadata):
```
Document: Nike Climate Impact Report 2025
Section: Environmental Commitments > Emission Reduction Goals
Type: Annual Report
Date: 2025-01-15

Nike has committed to reducing emissions by 50% by 2030...
```

**Full indexing implementation**:

```python
doc_title = "Nike Climate Impact Report 2025"
document_text = load_document("nike_climate_report.md")
chunks = create_contextual_chunks(document_text, doc_title, chunk_size=500)

contextualized_texts = [chunk['contextualized'] for chunk in chunks]
embeddings = model.encode(contextualized_texts)

for chunk, embedding in zip(chunks, embeddings):
    client.upsert(
        collection_name="documents",
        points=[PointStruct(
            id=chunk['metadata']['chunk_index'],
            vector=embedding.tolist(),
            payload={
                'contextualized_text': chunk['contextualized'],
                'original_text': chunk['original'],
                'doc_title': chunk['metadata']['doc_title'],
                'section': chunk['metadata']['section'],
                'chunk_index': chunk['metadata']['chunk_index'],
            }
        )]
    )

# At retrieval time: search uses contextualized embeddings, LLM receives original_text
query_results = client.search(collection_name="documents", query_vector=query_embedding, limit=5)
llm_context = [result.payload['original_text'] for result in query_results]
```

**Optimization**: Use abbreviated headers for frequently repeated metadata:

```python
# Standard (50 tokens)
"Document: Nike Climate Impact Report 2025\nSection: Environmental Commitments\n\n"

# Abbreviated (20 tokens)
"Doc: Nike Climate 2025 | Env Commitments\n\n"
```

</details>

**When to use**: Documents with clear hierarchy (reports, academic papers, legal docs), multi-document corpora, chunks that lack self-contained context. **When to skip**: Already self-contained documents (FAQs, product descriptions), single-document retrieval, or when simple chunking achieves acceptable quality.

**Trade-offs**: Dramatically improves retrieval precision and makes chunks self-contained. Costs 10-50 extra tokens per chunk for headers, increases embedding costs, and requires structure parsing. A/B test on your corpus before committing.

---

## Vector storage

You've chunked and embedded. Now you need a database optimized for "find the 5 closest vectors" across millions of chunks.

### In-memory vs persistent

**In-memory** (FAISS, Annoy): microsecond latency, no network overhead. Works for <10M chunks (~60GB at 1536 dims). Requires reloading on restart.

```python
import faiss
import numpy as np

dimension = 1536
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)
distances, indices = index.search(query_embedding[np.newaxis, :], k=5)
```

**Persistent** (Pinecone, Weaviate, Qdrant, Milvus): scales to billions, supports metadata filtering. Network latency (5-20ms) but handles larger-than-RAM datasets.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
client.upsert(collection_name="documents", points=points)
results = client.search(collection_name="documents", query_vector=query_embedding.tolist(), limit=5)
```

Use in-memory for <1M chunks, single-machine, <10ms latency requirements. Use persistent for >1M chunks, metadata filtering, multi-service access.

### Index types

| Algorithm | Recall@10 | Latency | Memory | When to use |
|-----------|-----------|---------|--------|-------------|
| **HNSW** | 95-99% | <10ms | High (~1.5x vectors) | Default. Best recall/latency trade-off. |
| **IVF** | 80-95% | <5ms | Low (1.5x vectors) | Memory-constrained. |
| **PQ** (Product Quantization) | 70-90% | <2ms | Very low (0.1x vectors) | Billion-scale compression. |
| **IVF+PQ** | 80-95% | <5ms | Low (0.2x vectors) | Billion-scale. Better recall than PQ alone. |
| **Flat** (brute-force) | 100% | Linear in N | 1x vectors | <10K chunks. Ground truth for ablations. |

HNSW is your default. Tune with `M=32, efConstruction=200, efSearch=50` for balanced performance (95% recall, <10ms). Increase for higher recall, decrease for speed.

**Recall@10**: probability that the true top-10 nearest neighbors appear in the approximate top-10. When you're retrieving 50 and reranking to 5, recall matters less — optimize for latency.

### Production patterns

1. **Build index offline**: Use a staging index, build fully, then hot-swap. Don't block queries during indexing.
2. **Shard by metadata**: Split by time, domain, or access pattern. Search shards selectively.
3. **Monitor recall**: Track both retrieval metrics (correct chunk in top-5?) and end-to-end metrics (LLM answered correctly?).
4. **Quantize**: Float16 halves memory with <1% quality loss. Int8 gives 4x compression with 5-10% loss.

---

## Chunking evaluation

The real goal is retrieval quality (did we find the right chunk?) and generation quality (did the LLM answer correctly?). But you can measure chunking quality independently.

### Intrinsic metrics (chunk quality)

**Chunk coherence**: Embed each sentence in a chunk, compute pairwise cosine similarity. High similarity = coherent. Target >0.7.

```python
def chunk_coherence(chunk: str) -> float:
    sentences = chunk.split('. ')
    if len(sentences) < 2:
        return 1.0
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    n = len(similarities)
    return (similarities.sum() - n) / (n * (n - 1))
```

**Boundary preservation**: Count splits that occur mid-sentence. Lower is better.

**Chunk size variance**: Coefficient of variation (std/mean). Target <0.3. High variance means some chunks are too small (no context) or too large (diluted relevance).

### Extrinsic metrics (retrieval quality)

Requires labeled query-document pairs.

**Hit Rate@k**: Fraction of queries where the correct chunk appears in top-k results. Target >0.9.

**MRR (Mean Reciprocal Rank)**: Average of 1/rank for the first correct chunk. 1.0 = always rank 1.

**NDCG@k**: Accounts for graded relevance. See [ML Evaluation]({{ site.baseurl }}/docs/machine-learning/evaluation/page/#workflow-ranking) for details.

### Evaluation workflow

1. **Create test set**: 100-500 queries with labeled correct chunks.
2. **Baseline**: Recursive splitting with default parameters.
3. **Ablations**: Vary chunk sizes (256, 500, 800), overlaps (0%, 10%, 20%), and strategies.
4. **Measure**: Hit Rate@5, MRR, chunk coherence per configuration.
5. **Select**: Maximize Hit Rate@5 subject to latency constraints. If two configs differ by <2%, pick the simpler one.

**Example results**:

| Configuration | Hit Rate@5 | MRR | Coherence | Indexing Time (1K docs) |
|---------------|------------|-----|-----------|-------------------------|
| Character split (baseline) | 0.65 | 0.42 | 0.55 | 5 min |
| Recursive (500 tokens, 10% overlap) | 0.82 | 0.61 | 0.74 | 8 min |
| Recursive (500 tokens, 20% overlap) | 0.85 | 0.65 | 0.78 | 10 min |
| Semantic (embedding-based) | 0.88 | 0.68 | 0.82 | 45 min |

Recursive (500 tokens, 20% overlap) wins. Semantic chunking adds 3% Hit Rate at 4.5x the cost — not worth it unless you're in a high-stakes regime. LLM-based semantic chunking is even slower (hours) with similar gains.

---

## Workflow: building an index

### Prototype pipeline

Get documents indexed and searchable in <50 lines.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

documents = ["RAG stands for Retrieval-Augmented Generation...", "..."]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=500, chunk_overlap=100
)
chunks = splitter.split_text("\n\n".join(documents))

model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
faiss.write_index(index, "index.faiss")
```

Exact search (IndexFlatL2) is fine for <10K chunks. No optimization, no metadata — but it works for prototypes and demos.

### Production pipeline

Production adds metadata tracking, incremental updates, ANN indexes, error handling, and monitoring.

```python
class ProductionIndexer:
    def __init__(self, collection_name="documents"):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=500, chunk_overlap=100
        )
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name

    def split(self, source: str, metadata: dict) -> list[str]:
        return self.splitter.split_text(source)

    def index_document(self, doc_id: str, text: str, metadata: dict) -> int:
        doc_hash = hashlib.sha256(text.encode()).hexdigest()

        # Skip if unchanged (idempotency)
        existing = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
            limit=1,
        )
        if existing[0] and existing[0][0].payload.get("doc_hash") == doc_hash:
            return 0

        # Delete old chunks, create new ones
        if existing[0]:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
            )

        chunks = self.split(text, metadata)
        embeddings = self.model.encode(chunks)
        points = [
            PointStruct(
                id=int(hashlib.sha256(f"{doc_id}-{i}".encode()).hexdigest()[:16], 16) % (2**63),
                vector=emb.tolist(),
                payload={"doc_id": doc_id, "doc_hash": doc_hash, "chunk_index": i, "text": chunk, **metadata},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(chunks)
```

<details class="boz-resource">
<summary>Batch indexing and multi-document type support</summary>

**Batch indexing**:

```python
def index_batch(self, documents: list[dict]) -> dict:
    stats = {"total_docs": len(documents), "total_chunks": 0, "failed": 0}
    for doc in documents:
        try:
            chunks = self.index_document(doc["doc_id"], doc["text"], doc.get("metadata", {}))
            stats["total_chunks"] += chunks
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Document {doc.get('doc_id')} failed: {e}")
    return stats
```

**Multi-document type indexer**: Route to appropriate splitters based on document type.

```python
class MultiTypeIndexer(ProductionIndexer):
    def __init__(self, collection_name="documents"):
        super().__init__(collection_name)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python", chunk_size=1000, chunk_overlap=200,
        )

    def split(self, source: str, metadata: dict) -> list[str]:
        doc_type = metadata.get("doc_type", "text")
        if doc_type == "markdown":
            return [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    for chunk in self.markdown_splitter.split_text(source)]
        elif doc_type == "code":
            return self.code_splitter.split_text(source)
        elif doc_type == "pdf":
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(source)
            markdown = result.document.export_to_markdown()
            return [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    for chunk in self.markdown_splitter.split_text(markdown)]
        else:
            return self.splitter.split_text(source)
```

Type-specific splitting prevents splitting mid-table (PDF), mid-function (code), or losing header context (markdown). Tag chunks with `doc_type` for type filtering at retrieval time.

</details>

Key additions over prototype: idempotency (doc_hash skips unchanged documents), change detection, metadata attachment, error handling, and logging.

### Incremental updates

Don't rebuild the entire index when you add 10 new documents. The `ProductionIndexer` above handles this: new documents are appended, updated documents have old chunks deleted and new ones inserted.

Rebuild only when parameters change: chunk size, embedding model, or index algorithm. Rebuilding a 1M chunk index takes hours; incremental updates take seconds.

---

## Common pitfalls

**Chunking without overlap**: If chunk N ends with "The solution is" and chunk N+1 starts with "to use caching", neither chunk answers "What's the solution?". Add 10-20% overlap.

**Using character count instead of token count**: A 1000-character chunk might be 250 or 400 tokens depending on content. Use `from_tiktoken_encoder` or count tokens explicitly with `tiktoken`.

**Ignoring chunk size distribution**: If 10% of chunks are <100 tokens or >2000 tokens, your splitting produces outliers that hurt retrieval.

**Reindexing everything on every change**: Incremental updates are 100x faster. Only rebuild when parameters change.

**Embedding before chunking**: Chunk first, embed chunks. You can't split a vector without re-embedding.

**Losing document metadata**: If you only store chunk text, you can't filter by author, date, or category. Always attach metadata.

**Mixing incompatible embeddings**: Switching models requires full reindex. New and old embeddings live in different spaces.

**Skipping evaluation**: You don't know if recursive splitting beats semantic chunking until you measure Hit Rate@5. Evaluate every strategy change.

**Chunking code like prose**: Use code-aware splitters. A chunk containing `def foo(x:` without the closing `)` confuses retrieval.

**Trusting layout-agnostic PDF extraction**: `pypdf`'s `PdfReader(...).pages[i].extract_text()` loses all layout. Tables become unreadable. Use layout-aware parsers like Docling, LlamaParse, or Marker.

**Hardcoding chunk size**: Different use cases need different sizes. FAQ queries want 500 tokens for precision; explanations want 1500 for context.

**Not monitoring index growth**: A document updated 100 times creates 100x chunks without deletion. Implement TTL or version limits.

---

## Operations & planning

RAG indexing has real costs — embedding API calls, storage, compute, and team time. These sections provide concrete numbers for budgeting, vendor selection, and staffing.

<details class="boz-resource">
<summary>Cost planning & budgeting</summary>

### Indexing costs

**Embedding API costs** scale linearly with token count. Check your provider's current per-token rate and multiply:

`total_cost = (doc_count × avg_tokens_per_doc) × price_per_token`

| Document Count | Avg Tokens/Doc | Total Tokens |
|----------------|----------------|--------------|
| 10,000 docs | 500 | 5M tokens |
| 100,000 docs | 500 | 50M tokens |
| 1,000,000 docs | 500 | 500M tokens |
| 1,000,000 docs | 2000 (long docs) | 2B tokens |

Most providers offer a **batch tier** (delayed processing window) at ~50% discount. Use batch for initial indexing, standard for incremental updates.

**Cost optimization**: Batch documents per API call, use the batch tier for bulk indexing, and consider self-hosted models (e.g., SentenceTransformers on GPU) when API spend dominates and you already have GPU/ops capacity.

### Storage costs

**Storage calculation**: `num_chunks × (vector_dim × 4 bytes + metadata_size)`. Example: 1M chunks, 768-dim, 500 bytes metadata = 3.57 GB.

| Provider type | Relative cost | Notes |
|---------------|---------------|-------|
| **Managed SaaS** (Pinecone, Weaviate Cloud) | $$$ | Zero ops, pay-per-use or monthly minimum, fastest TTM |
| **Hybrid cloud** (Qdrant Cloud, Weaviate Plus) | $$ | Your infra + managed control plane, some free tiers available |
| **Self-hosted** (Qdrant, Milvus) | $ | Full control, requires DevOps (0.5+ FTE) |
| **In-memory** (FAISS) | Server cost only | Not persistent, prototype/research use, <10M chunks |

Always use each vendor's **pricing calculator** for accurate estimates — pricing models vary (per-GB, per-vector-dimension, per-read-unit) and change frequently.

**Optimization**: Quantize to float16 (50% storage savings, <1% quality loss), implement document TTL, monitor metadata size.

### Query costs

Query costs have three components, ranked by typical magnitude:

1. **Reranking** (cross-encoder, GPU) — most expensive per query, 10-100× embedding cost
2. **Vector search** (managed) — included in plan or usage-based; self-hosted is near-zero marginal cost
3. **Query embedding** — negligible; a typical 30-50 token query costs a fraction of a cent per 1K queries at current API rates

**Rule of thumb**: At low-to-moderate query volumes (<100K/day), embedding and search costs are dwarfed by the fixed monthly cost of your vector database. Reranking is the main variable cost to watch.

### Total cost of ownership

| Component | Managed SaaS | Hybrid Cloud | Self-Hosted |
|-----------|-------------|--------------|-------------|
| Infra cost | Highest | Medium | Lowest |
| Query cost | Usage-based | Usage-based | Near-zero marginal |
| Ops overhead | ~0 FTE | ~0.1 FTE | 0.5+ FTE |
| Time to production | Weeks | Weeks | Months |

**When does self-hosted break even?** Only when infrastructure savings exceed the cost of ops labor. Typically this requires >100K queries/day or >5M chunks and a team that already has Kubernetes/DevOps capacity. Below that, managed services save money because labor cost dominates infrastructure cost.

**Verify current pricing**: [OpenAI Embeddings](https://platform.openai.com/docs/pricing), [Pinecone](https://www.pinecone.io/pricing/), [Qdrant](https://qdrant.tech/pricing/), [Weaviate](https://weaviate.io/pricing)

</details>

<details class="boz-resource">
<summary>Build vs buy decision framework</summary>

### Comparison matrix

| Factor | Self-Hosted (Qdrant/Milvus) | Managed SaaS (Pinecone/Weaviate Cloud) |
|--------|------------------------------|----------------------------------------|
| Relative infra cost | Lower (server/VM costs only) | Higher (convenience premium) |
| Time to production | 8-12 weeks | 2-4 weeks |
| Engineering overhead | 0.5-1 FTE ops | ~0 FTE |
| Control & customization | Full | Limited |
| Vendor lock-in | None | High |
| Compliance | DIY | Vendor-certified (SOC 2, HIPAA BAA) |

### Vendor landscape

| Vendor | Best for | Pricing model |
|--------|----------|---------------|
| **Pinecone** | Fast TTM, small teams, SOC 2/HIPAA | Per-GB storage + per-read/write unit |
| **Weaviate Cloud** | GraphQL, multi-modal, open-source preference | Per-vector-dimension + storage |
| **Qdrant Cloud** | Cost-conscious, RBAC filtering, hybrid deploy | Resource-based (RAM/CPU/storage), free tier available |
| **Qdrant Self-Hosted** | Scale optimization, full control, data sovereignty | Server cost + ops labor |
| **Milvus Self-Hosted** | Billion-scale, GPU acceleration, research | Server cost + 1-2 FTE ops |
| **pgvector** (PostgreSQL) | Teams already running Postgres, avoiding new infra | Included in Postgres hosting cost |
| **ChromaDB** | Fast RAG prototyping, Python-native, embedded use | Free (open-source), cloud option emerging |
| **FAISS In-Memory** | Benchmarking, research, <1M chunks | Server cost only |

</details>

---

## Best practices

**Start with recursive splitting.** 500-token chunks, 100-token overlap, default separators. Works for most use cases. Move to smarter splitting only when you've measured that chunking is the bottleneck.

**Measure retrieval quality, not generation quality, when tuning chunking.** If Hit Rate@5 is 90% but the LLM hallucinates, chunking is not the problem. Track MRR and NDCG@k alongside Hit Rate for a complete picture.

**Overlap usually helps.** 10-20% prevents information loss at boundaries. Test with your retriever — some sparse models (e.g. SPLADE) show no benefit.

**Use metadata for filtering.** Attach structure (sections, timestamps, authors) to chunks. Filter at retrieval time — reduces search space and improves relevance compared to post-processing.

**Don't chunk code like prose.** Use code-aware splitters that respect syntax boundaries.

**PDF structure matters.** Use layout-aware parsers (Docling, LlamaParse, Marker) that preserve tables and section boundaries.

**Quantize embeddings in production.** Float16 halves storage with <1% quality loss.

**Reindex incrementally.** Full reindex only for parameter changes, not content updates.

**Monitor chunk size distribution.** Outliers (<100 or >2000 tokens) hurt retrieval.

**Implement RBAC metadata for enterprise.** Add roles, departments, owner_id at indexing time for hard security filtering at retrieval time without reindexing. For complex hierarchies (users in hundreds of groups), consider post-retrieval authorization instead of metadata explosion.

**Consider parent-child indexing.** Embed small chunks for precision, but return the parent section for context. Balances retrieval accuracy with LLM context needs.

**Index for hybrid search.** Maintain both vector and keyword (BM25) indexes. Dense + sparse retrieval consistently outperforms either alone, especially for exact-match terms. Vector databases like Qdrant and Weaviate support sparse vectors natively; for others, maintain a separate BM25 index (e.g. Elasticsearch, tantivy).

**Test on real queries.** Sample 500 production queries, label correct chunks, measure Hit Rate@5. This is your ground truth.
