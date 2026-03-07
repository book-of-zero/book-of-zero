---
layout: post
title: "RAG Retrieval: vector search and query strategies"
nav_order: 6
---

You indexed a million documents. The user asks a question. You have 100ms to find the 5 most relevant chunks before the LLM generates an answer. Naive search fails — keyword mismatch, semantic drift, or sheer scale. The solution: a multi-stage retrieval pipeline.

Fast retrieval uses two-tower architectures and approximate nearest neighbor search to find 50-100 candidates in <50ms. Precise reranking uses cross-encoders to refine the top candidates in another <50ms. Together they give you the best of both worlds: speed and accuracy.

---

## On this page

- [Key concepts](#key-concepts)
- [Query routing & intent classification](#query-routing--intent-classification)
- [Stage 1: Fast retrieval](#stage-1-fast-retrieval)
  - [Two-tower architecture](#two-tower-architecture)
  - [Matryoshka embeddings (variable-dimension)](#matryoshka-embeddings-variable-dimension)
  - [BM25 sparse retrieval](#bm25-sparse-retrieval)
  - [Learned sparse retrieval (SPLADE)](#learned-sparse-retrieval-splade)
  - [Hybrid search](#hybrid-search)
- [Stage 2: Reranking](#stage-2-reranking)
  - [Cross-encoder architecture](#cross-encoder-architecture)
  - [ColBERT (Late Interaction)](#colbert-late-interaction)
  - [LLM-based rerankers](#llm-based-rerankers)
- [Search-time index tuning](#search-time-index-tuning)
- [Search-time filtering & boosting](#search-time-filtering--boosting)
  - [Metadata filtering](#metadata-filtering)
  - [Hard security filtering (RBAC)](#hard-security-filtering-rbac)
  - [Score boosting](#score-boosting)
  - [Positional boosting](#positional-boosting)
- [Query optimization](#query-optimization)
  - [Query expansion](#query-expansion)
  - [Conversational query rewriting](#conversational-query-rewriting)
  - [Query decomposition](#query-decomposition)
  - [Self-querying (natural language to metadata filters)](#self-querying-natural-language-to-metadata-filters)
  - [Advanced query techniques](#advanced-query-techniques)
- [Result optimization](#result-optimization)
  - [MMR (Maximal Marginal Relevance)](#mmr-maximal-marginal-relevance)
  - [Contextual compression](#contextual-compression)
  - [Parent-child retrieval](#parent-child-retrieval)
  - [Document-level vs chunk-level retrieval](#document-level-vs-chunk-level-retrieval)
  - [Context enrichment](#context-enrichment)
- [Semantic caching](#semantic-caching)
- [Workflow: building a retrieval pipeline](#workflow-building-a-retrieval-pipeline)
- [Agentic retrieval](#agentic-retrieval)
  - [Pipeline vs agentic RAG](#pipeline-vs-agentic-rag)
  - [Key agentic patterns](#key-agentic-patterns)
- [Beyond text: emerging retrieval paradigms](#beyond-text-emerging-retrieval-paradigms)
  - [GraphRAG (Knowledge Graph Retrieval)](#graphrag-knowledge-graph-retrieval)
  - [Multimodal retrieval](#multimodal-retrieval)
- [Common pitfalls](#common-pitfalls)
- [Production quality metrics & SLAs](#production-quality-metrics--slas)
  - [Retrieval quality benchmarks](#retrieval-quality-benchmarks)
  - [Synthetic test data & automated evaluation](#synthetic-test-data--automated-evaluation)
  - [Latency SLAs](#latency-slas)
  - [User satisfaction metrics](#user-satisfaction-metrics)
  - [Monitoring dashboard](#monitoring-dashboard)
  - [Quality vs cost trade-offs](#quality-vs-cost-trade-offs)
- [Rollout & change management](#rollout--change-management)
  - [Phased rollout strategy](#phased-rollout-strategy)
  - [A/B testing framework](#ab-testing-framework)
  - [Rollback plan](#rollback-plan)
  - [Change management](#change-management)
- [Operational risks & mitigation](#operational-risks--mitigation)
  - [Infrastructure risks](#infrastructure-risks)
  - [Quality risks](#quality-risks)
  - [Security risks](#security-risks)
  - [Cost risks](#cost-risks)
  - [Business continuity](#business-continuity)
- [Best practices](#best-practices)

---

## Key concepts

- **Two-tower architecture**: Queries and documents encoded independently by the same model into vectors compared via cosine similarity. Documents are embedded offline; only the query is embedded at search time.
- **Dense retrieval**: Semantic search using learned embeddings. Captures meaning but misses exact keyword matches. DPR is the landmark implementation.
- **Sparse retrieval**: Keyword-based search using term frequencies (BM25). Fast and interpretable but fails on semantic paraphrases.
- **Learned sparse retrieval**: Models like SPLADE that use transformers to predict term importance weights across the vocabulary, producing sparse vectors that capture semantic meaning while remaining compatible with inverted index infrastructure.
- **Hybrid search**: Combining dense and sparse retrieval with score normalization and reranking. Typically improves Hit Rate@10 by 5-15% over either method alone.
- **Cross-encoder**: Jointly encodes query-document pairs for reranking. More accurate than two-tower but ~100x slower since it cannot precompute document representations.
- **LLM-based reranker**: Uses large language models (Cohere Rerank, RankLLaMA, RankGPT) for reranking. Achieves 2-5% higher NDCG@10 than cross-encoders on diverse benchmarks but at higher latency and cost.
- **ColBERT**: Late interaction architecture encoding queries and documents as token-level embeddings (N tokens x 128-dim) with MaxSim scoring. Achieves 95-98% of cross-encoder accuracy at 10x lower latency.
- **Matryoshka embeddings (MRL)**: Embedding models trained so that the first *d* dimensions form a valid lower-dimensional embedding. Truncate 3072-dim to 256-dim with minimal accuracy drop, making ANN search up to 12x cheaper. Supported by OpenAI v3, Nomic Embed v1.5, Jina v3, Cohere Embed v4.
- **Self-querying**: Extracting structured metadata filters from natural language queries using an LLM or NER model. "Healthcare policies from 2023" → semantic query "policies" + filter {category: "healthcare", year: >= 2023}.
- **Agentic retrieval**: Replacing fixed retrieval pipelines with an LLM agent that has retrieval as a tool. The agent decides when and how often to search, enabling iterative multi-hop retrieval at the cost of variable and higher latency.
- **ANN (Approximate Nearest Neighbor)**: Algorithms (HNSW, IVF, PQ) that trade exact accuracy for sublinear search speed. Required past ~10K documents.
- **RBAC (Role-Based Access Control)**: Security filtering that physically blocks unauthorized chunks at query time via metadata filters. Distinct from soft boosting which only adjusts scores.
- **Semantic caching**: Caching retrieval results by query similarity (cosine >= 0.90-0.95) rather than exact string match. Achieves significantly higher hit rates than exact-match caching, especially in FAQ/support systems.

---

## Query routing & intent classification

Not all queries should go to the vector database. Some need SQL, some are greetings, some are commands. **Query routing** classifies query intent and routes to the appropriate backend. Skip this step and your RAG system will waste time retrieving documents for "Hello" or "What's my account balance?" (which needs a database query, not document search).

Classify queries into predefined intents, each mapped to a backend:

| Intent | Backend | Example Queries |
|--------|---------|-----------------|
| **Greeting** | Direct response | "Hi", "Thanks", "Goodbye" |
| **Knowledge base** | Vector DB (RAG) | "How do I reset password?", "What is RAG?" |
| **Structured data** | SQL database | "What's my balance?", "Show my orders" |
| **Real-time data** | External API | "What's the weather?", "AAPL stock price" |
| **Out of scope** | Direct response (apologize) | "Tell me a joke", "Write code for me" |

Four approaches to intent classification: rule-based (keyword matching, <1ms, for <10 intents), ML-based (embed queries + train classifier, for 10+ intents), semantic similarity (few-shot matching against intent exemplars, no training needed), and LLM-based (structured output from an LLM call, most flexible but slowest at 200-1000ms).

```python
from sentence_transformers import SentenceTransformer, util

ENCODER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INTENT = "knowledge_base"

INTENT_EXAMPLES: dict[str, list[str]] = {
    "greeting": ["Hi", "Hello", "Thanks", "Goodbye"],
    "knowledge_base": ["How do I reset my password?", "What is RAG?"],
    "structured_data": ["What's my account balance?", "Show my recent orders"],
    "real_time": ["What's the weather in SF?", "AAPL stock price"],
    "out_of_scope": ["Tell me a joke", "What's the meaning of life?"],
}


class SemanticIntentClassifier:
    """Classifies query intent by cosine similarity against few-shot exemplars."""

    def __init__(self, model_name: str = ENCODER_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
        self.intent_embeddings = {
            intent: self.model.encode(examples)
            for intent, examples in INTENT_EXAMPLES.items()
        }

    def classify(self, query: str, threshold: float = 0.5) -> str:
        """Returns the best-matching intent, or DEFAULT_INTENT if below threshold."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        query_embedding = self.model.encode(query)
        intent_scores = {
            intent: float(util.cos_sim(query_embedding, embs)[0].max())
            for intent, embs in self.intent_embeddings.items()
        }
        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent if intent_scores[best_intent] >= threshold else DEFAULT_INTENT
```

**Rule-based (for small systems with <10 intents):**

```python
import re
from typing import Literal

IntentType = Literal["greeting", "knowledge_base", "structured_data", "real_time", "out_of_scope"]

INTENT_PATTERNS: dict[IntentType, str] = {
    "greeting": r"\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b",
    "structured_data": r"\b(my account|my balance|my order|my subscription|purchase history)\b",
    "real_time": r"\b(weather|temperature|forecast|stock price|news)\b",
    "out_of_scope": r"\b(joke|poem|story|write code|meaning of life)\b",
}


def classify_intent_simple(query: str) -> IntentType:
    """Matches query against keyword patterns; falls back to knowledge_base."""
    query_lower = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, query_lower):
            return intent
    return "knowledge_base"
```

**ML-based (for 10+ intents or nuanced queries):**

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ENCODER_MODEL = "all-MiniLM-L6-v2"
MAX_SOLVER_ITERATIONS = 1000


class MLIntentClassifier:
    """Trains a logistic regression classifier on embedded query examples."""

    def __init__(
        self,
        training_data: list[tuple[str, str]],
        model_name: str = ENCODER_MODEL,
    ) -> None:
        self.encoder = SentenceTransformer(model_name)
        queries, intents = zip(*training_data)
        embeddings = self.encoder.encode(queries)
        self.classifier = LogisticRegression(max_iter=MAX_SOLVER_ITERATIONS)
        self.classifier.fit(embeddings, intents)

    def classify(self, query: str) -> str:
        """Returns predicted intent label for a single query."""
        return self.classifier.predict([self.encoder.encode([query])[0]])[0]


training_data = [
    ("Hi", "greeting"), ("How do I reset my password?", "knowledge_base"),
    ("What's my account balance?", "structured_data"),
    ("What's the weather in SF?", "real_time"),
    ("Tell me a joke", "out_of_scope"),
    # ... 100-1000+ examples
]
classifier = MLIntentClassifier(training_data)
```

**Routing logic:**

```python
GREETING_RESPONSE = "Hi! How can I help?"
OUT_OF_SCOPE_RESPONSE = "I'm designed for account and product support."


def route_query(query: str, user_id: str) -> dict[str, str]:
    """Classifies query intent and dispatches to the appropriate backend."""
    intent = classify_intent(query)
    if intent == "greeting":
        return {"intent": intent, "response": GREETING_RESPONSE, "source": "direct"}
    if intent == "knowledge_base":
        chunks = hybrid_search(query, documents, k=5)
        return {"intent": intent, "response": llm_generate(query, chunks), "source": "vector_db"}
    if intent == "structured_data":
        result = database.execute(generate_sql(query, user_id))
        return {"intent": intent, "response": format_sql_result(result), "source": "sql"}
    if intent == "real_time":
        return {"intent": intent, "response": call_api(query), "source": "api"}
    return {"intent": "out_of_scope", "response": OUT_OF_SCOPE_RESPONSE, "source": "direct"}
```

**Multi-step routing** for complex queries needing multiple backends (e.g., "Show my recent orders and recommend similar products"): use an LLM to decompose the query into steps, classify and route each step independently, then combine results.

**When routing matters**: Multi-backend systems (RAG + SQL + APIs), customer support chatbots, enterprise assistants. Less important for single-backend systems or specialized tools. Cost: 5-20ms latency with local embeddings (semantic), 50-200ms with API-based embeddings, or <1ms (rule-based) — routing avoids up to 30-60% of unnecessary retrievals.

---

## Stage 1: Fast retrieval

Stage 1 retrieves 50-100 candidate chunks from millions in <50ms. The goal is high recall — you want the correct chunk in the top 50 even if it's not ranked 1st. Precision comes in stage 2.

### Two-tower architecture

The two-tower architecture is why dense retrieval scales. You encode queries and documents separately with the same encoder, then compare via cosine similarity.

```
┌─────────────┐          ┌─────────────┐
│   Query     │          │  Document   │
└──────┬──────┘          └──────┬──────┘
       │  Encoder (shared)      │  Encoder (shared)
       ▼                        ▼
  [ 0.2, 0.8, ... ]        [ 0.3, 0.7, ... ]
       Query vector            Doc vector
              └────────┬───────────┘
                       ▼
                Cosine similarity
```

Documents are encoded once at index time. At search time, you only encode the query (5-50ms) then compute cosine similarity against precomputed document vectors (<1ms with ANN).

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
HNSW_M = 32
STAGE1_K = 50

model = SentenceTransformer(ENCODER_MODEL)

doc_embeddings = model.encode(documents, convert_to_numpy=True)
# L2 default; equivalent to cosine on normalized vectors
index = faiss.IndexHNSWFlat(doc_embeddings.shape[1], HNSW_M)
index.add(doc_embeddings)

query_embedding = model.encode("What is RAG?", convert_to_numpy=True)
distances, indices = index.search(query_embedding[np.newaxis, :], k=STAGE1_K)
```

The landmark two-tower architecture from Facebook AI Research. Uses separate BERT-based encoders for queries and contexts, trained with contrastive loss.

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import numpy as np

DPR_QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
DPR_CONTEXT_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
STAGE1_K = 50

q_encoder = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_MODEL)
c_encoder = DPRContextEncoder.from_pretrained(DPR_CONTEXT_MODEL)
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_QUESTION_MODEL)
c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(DPR_CONTEXT_MODEL)

doc_tokens = c_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    doc_embeddings = c_encoder(**doc_tokens).pooler_output.numpy()

query_tokens = q_tokenizer([query], return_tensors="pt")
with torch.no_grad():
    query_embedding = q_encoder(**query_tokens).pooler_output.numpy()

similarities = np.dot(doc_embeddings, query_embedding.T).squeeze()
top_indices = np.argsort(similarities)[::-1][:STAGE1_K]
```

**When two-tower works**: Semantic paraphrases — "What causes rain?" matches "Precipitation occurs when atmospheric water vapor condenses..." despite sharing no keywords.

**When two-tower fails**: Exact keyword matches matter. "HIPAA compliance requirements" should strongly match documents containing "HIPAA." Solution: hybrid search.

### Matryoshka embeddings (variable-dimension)

Standard embedding models produce fixed-dimension vectors (768-dim, 1536-dim). **Matryoshka Representation Learning** (MRL, Kusupati et al., NeurIPS 2022) trains models so that the first *d* dimensions of the full vector form a valid, lower-dimensional embedding. You can truncate a 3072-dim vector to 256 dimensions with only a small accuracy drop — making ANN search, memory, and storage up to 12x cheaper.

| Model | Full dims (MTEB) | Truncated dims | Performance note | Storage savings |
|-------|------------------|---------------|------------------|-----------------|
| **OpenAI text-embedding-3-large** | 3072 (64.6%) | 256 | Still outperforms ada-002 (61.0% at 1536d) | 12x |
| **OpenAI text-embedding-3-large** | 3072 (64.6%) | 1024 | Near-identical to 3072 (sweet spot) | 3x |
| **OpenAI text-embedding-3-small** | 1536 (62.3%) | 512 | ~1-2% drop | 3x |
| **Nomic Embed v1.5** | 768 (62.3%) | 256 | Outperforms ada-002 at 512d | 3x |
| **Jina Embeddings v3** | 1024 | 64 | Retains ~92% retrieval performance | 16x |
| **Cohere Embed v4** | 1536 | 256 | MRL supported (v3 does NOT support truncation) | 6x |

```python
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"
TRUNCATED_DIMS = 256

client = OpenAI()

full = client.embeddings.create(input="What is RAG?", model=EMBEDDING_MODEL)

compact = client.embeddings.create(
    input="What is RAG?", model=EMBEDDING_MODEL, dimensions=TRUNCATED_DIMS
)
# For models without native truncation: embedding[:TRUNCATED_DIMS] then L2-normalize
```

**MRL in two-stage retrieval**: Use 256-dim embeddings for stage 1 (fast ANN recall over millions of documents), then let the cross-encoder handle precision in stage 2. The accuracy drop from truncation is typically smaller than the cross-encoder's improvement — so the pipeline as a whole loses almost nothing while stage 1 becomes significantly faster and cheaper.

**Vector DB support**: Milvus (v2.5+ "Funnel Search" — shortlist with truncated dims, rescore with full dims in one query), Qdrant (prefetch + rescore via Query API), Weaviate (configurable `dimensions` for OpenAI, named vectors for multi-resolution). Pinecone, pgvector, Chroma accept any dimension but require application-level two-stage logic.

**When MRL helps**: Large-scale indexes (>1M documents) where storage and ANN latency matter, or when you want to reduce embedding API costs. **When to skip**: Small indexes where full-dimension search is already fast enough, or when you need maximum stage 1 precision without reranking.

### BM25 sparse retrieval

BM25 scores documents by term frequency (TF) and inverse document frequency (IDF), with saturation to prevent over-weighting of repeated terms.

```python
from rank_bm25 import BM25Okapi
import numpy as np

STAGE1_K = 50

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

query = "HIPAA compliance requirements"
scores = bm25.get_scores(query.lower().split())
top_indices = np.argsort(scores)[::-1][:STAGE1_K]
```

**How BM25 works**: TF rewards more term occurrences with saturation (10th occurrence adds less than 1st). IDF weights rare terms higher ("HIPAA" beats "the"). Document length normalization prevents unfair penalization of long documents. Key parameters: k1 (term saturation, default 1.2) and b (length normalization, default 0.75).

**When BM25 works**: Exact keyword matches, named entities, acronyms. **When BM25 fails**: Semantic paraphrases.

### Learned sparse retrieval (SPLADE)

BM25 relies on exact term matching and hand-crafted statistics. **SPLADE** (SParse Lexical AnD Expansion) uses a transformer (typically DistilBERT or BERT) to predict importance weights for every term in the vocabulary, producing a sparse vector where most weights are zero. This gives BM25-style efficiency with dense retrieval-style semantic understanding.

```
BM25:    "machine learning" → weights only "machine" and "learning"
SPLADE:  "machine learning" → weights "machine", "learning", "ML", "algorithm",
          "neural", "model", "AI", "training" (learned expansions)
```

SPLADE solves the vocabulary mismatch problem that plagues BM25 — a query about "ML pipelines" can match documents containing "machine learning workflows" because the model learns to expand both into overlapping terms.

| Model | MRR@10 (MS MARCO dev) | Sparsity | Query Latency |
|-------|----------------------|----------|---------------|
| **BM25** | 18.7% | Very high | <5ms |
| **SPLADE** (original) | 32.8% | Medium | 30-50ms |
| **SPLADEv2** | 36.8% | High | 30-45ms |
| **SPLADE++ (EnsembleDistil)** | 38.3% | High | 30-45ms |
| **SPLADE-v3** | 40.2% | High | 30-45ms |

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
MAX_TOKEN_LENGTH = 256


class SpladeEncoder:
    """Encodes text into SPLADE sparse vectors using learned term expansion."""

    def __init__(self, model_name: str = SPLADE_MODEL) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def encode(self, text: str) -> dict[str, float]:
        """Returns sparse vector mapping vocabulary terms to importance weights."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH)
        with torch.no_grad():
            logits = self.model(**tokens).logits
        # Max pooling over sequence length, then ReLU + log(1+x) for sparsity
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1
        ).values
        non_zero = weights.squeeze().nonzero().squeeze()
        return {self.tokenizer.decode(idx.item()): weights[0, idx].item() for idx in non_zero}
```

**When SPLADE works**: Hybrid setups where you want semantic sparse retrieval without a separate dense index. Particularly strong for domain-specific vocabulary and abbreviations. **When to skip**: If you already run hybrid dense + BM25 and don't want to retrain; or if sub-5ms latency is required (BM25 is faster).

**Infrastructure**: SPLADE outputs are sparse vectors compatible with inverted indexes. Qdrant (native sparse vectors + built-in SPLADE via FastEmbed), Vespa (native SPLADE embedder), Milvus (sparse vectors in same collection as dense), and Pinecone (sparse-dense hybrid) all support sparse vector search. Weaviate uses BM25 for its sparse component and does not natively support SPLADE vectors.

### Hybrid search

Combine dense and sparse retrieval to get the best of both: semantic understanding and keyword precision.

```python
EPSILON = 1e-8


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalizes scores to [0, 1] range."""
    return (scores - scores.min()) / (scores.max() - scores.min() + EPSILON)


def hybrid_search(
    query: str,
    documents: list[str],
    doc_embeddings: np.ndarray,
    encoder: SentenceTransformer,
    bm25: BM25Okapi,
    k: int = 50,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Fuses dense cosine and BM25 sparse scores with min-max normalization."""
    if not documents:
        raise ValueError("documents must be non-empty")

    query_embedding = encoder.encode(query, convert_to_numpy=True)
    dense_scores = np.dot(doc_embeddings, query_embedding)
    sparse_scores = bm25.get_scores(query.lower().split())

    # Normalize both to [0, 1] — critical for fair combination
    dense_norm = _min_max_normalize(dense_scores)
    sparse_norm = _min_max_normalize(sparse_scores)

    hybrid_scores = alpha * dense_norm + (1 - alpha) * sparse_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    return [(int(i), float(hybrid_scores[i])) for i in top_indices]
```

Precompute `doc_embeddings` once at index time. Search-time work should encode only the query, score against stored vectors, and fuse with sparse scores.

**Tuning alpha**: alpha=1.0 pure dense, alpha=0.0 pure sparse, alpha=0.5 balanced default, alpha=0.7 favor semantic (conversational), alpha=0.3 favor keywords (technical/entity search). Hybrid typically improves Hit Rate@10 by 5-15% over single-method, depending on dataset and metric.

**Alternative: Reciprocal Rank Fusion (RRF)** merges ranked lists without score normalization: `score(doc) = sum(1 / (k + rank))` where rank starts at 1 and k=60. More robust than score combination but less tunable. Use RRF when you don't have time to tune alpha.

```python
def reciprocal_rank_fusion(
    dense_ranks: list[int],
    sparse_ranks: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merges ranked lists using RRF: score(doc) = sum(1 / (k + rank))."""
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(dense_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## Stage 2: Reranking

Stage 1 gave you 50 candidates. Stage 2 refines them to the top 5-10 using a more accurate but slower model.

### Cross-encoder architecture

Unlike two-tower (separate encoders), cross-encoders jointly encode query and document. The model sees attention between query and document tokens — "capital" in the query connects to "capital" and "Paris" in the document. This captures interaction that two-tower misses but requires encoding every query-document pair (no precomputation).

```python
from sentence_transformers import CrossEncoder
import numpy as np

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
STAGE1_K = 50
FINAL_K = 5

reranker = CrossEncoder(RERANKER_MODEL)

candidates = hybrid_search(query, documents, k=STAGE1_K)
candidate_docs = [documents[i] for i, score in candidates]

pairs = [[query, doc] for doc in candidate_docs]
rerank_scores = reranker.predict(pairs)
top_docs = [candidate_docs[i] for i in np.argsort(rerank_scores)[::-1][:FINAL_K]]
```

**Models**: ms-marco-MiniLM-L6-v2 (fast, 6 layers, default), ms-marco-MiniLM-L12-v2 (better quality, 2x slower), mmarco-mMiniLMv2-L12-H384-v1 (multilingual, 14 languages).

**When reranking matters**: Stage 1 has high recall but low precision (correct chunk in top 50 but ranked 20th), or queries require nuanced understanding ("compare X and Y"). Typically yields a meaningful MRR improvement, especially when stage 1 recall is high but precision is low. **Skip reranking** when latency budget is very tight or stage 1 already achieves high Hit Rate@5.

**Latency**: Cross-encoder latency scales with model size and candidate count. A small cross-encoder (MiniLM-L6) reranking 50 candidates on GPU runs in tens of milliseconds; larger models or hundreds of candidates can push into hundreds of milliseconds. Budget accordingly — a two-stage pipeline with a small cross-encoder reranking 50 candidates typically adds minimal overhead on top of stage 1.

### ColBERT (Late Interaction)

ColBERT bridges the accuracy-latency gap between two-tower and cross-encoder. It encodes queries and documents independently at **token-level granularity**, then computes similarity via **MaxSim** (for each query token, find max similarity with any doc token, sum the scores).

```
Two-tower:  Query → [single 768-dim vector], Doc → [single 768-dim vector]
ColBERT:    Query → [N × 128-dim vectors],  Doc → [M × 128-dim vectors]
            Similarity = Σ max(cos(Qi, Dj)) for all query tokens Qi
```

| Architecture | Candidates | Relative latency | MRR@10 | Storage |
|--------------|------------|------------------|---------|---------|
| **Bi-encoder** | 1M | Fastest | 35-37% | 1x (768-dim/doc) |
| **ColBERT** | 100-1000 | ~10x bi-encoder | 39.7% | 6-10x (128-dim/token) |
| **Cross-encoder** | 10-50 | ~100x bi-encoder (model & k dependent) | 42-44% | Model params only |

```python
from ragatouille import RAGPretrainedModel

COLBERT_MODEL = "colbert-ir/colbertv2.0"
STAGE1_K = 100
STAGE2_K = 50
FINAL_K = 5

colbert_model = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)

stage1_results = dense_retrieval(query, k=STAGE1_K)
stage2_results = colbert_model.rerank(query=query, documents=stage1_results, k=STAGE2_K)
stage3_results = cross_encoder.rerank(query, stage2_results, k=FINAL_K)
```

**Use ColBERT** when reranking 100-1000 candidates where cross-encoder is too slow. **Skip ColBERT** for <100 candidates (cross-encoder is fast enough) or when 6-10x storage overhead is unacceptable.

**Compression**: ColBERTv2 uses centroid-based residual compression (6-10x smaller than v1). PLAID accelerates retrieval via centroid interaction and pruning (2.5-6.8x speedup, <1% accuracy loss). WARP further reduces index size 2-4x over PLAID via implicit decompression and dynamic similarity imputation.

### LLM-based rerankers

Cross-encoders use BERT-scale models (110M-340M parameters). LLM-based rerankers use larger language models for higher accuracy at the cost of latency and compute.

| Reranker | Approach | NDCG@10 | Cost |
|----------|----------|---------|------|
| **ms-marco-MiniLM-L6-v2** | Cross-encoder (22M) | ~49% (BEIR avg) | Self-hosted |
| **BGE Reranker v2-m3** | Cross-encoder (0.6B) | ~66% (BEIR avg) | Self-hosted |
| **Cohere Rerank 3.5** | Purpose-built (API) | ~56% (BEIR avg) | API (pay-per-search) |
| **RankLLaMA-13B** | Fine-tuned LLaMA | ~76% (DL19) | Self-hosted (GPU) |
| **RankGPT (GPT-4)** | Listwise via LLM | ~75% (DL19) | API (high) |

**How they differ**: Cross-encoders (MiniLM, BGE) score each query-document pair independently (pointwise) and are purpose-built for relevance scoring — significantly cheaper and faster than general-purpose LLM rerankers. RankLLaMA fine-tunes a 13B LLM for pointwise scoring. RankGPT passes all candidates to GPT-4 and asks it to rank them as a list (listwise), achieving the highest accuracy but at significant cost.

**When to use LLM-based rerankers**: High-stakes domains (legal, medical, compliance) where accuracy improvement justifies the latency and cost. For most production systems, purpose-built rerankers (BGE, Cohere) offer the best price-performance ratio. **When to use RankGPT/RankLLaMA**: Research, low-volume high-stakes applications, or when cross-encoder accuracy is insufficient. **When to skip entirely**: Latency budget <100ms or cost-sensitive workloads.

---

## Search-time index tuning

Vector indexes are built during indexing (see [RAG Indexing Vector Storage]({{ site.baseurl }}/docs/genai/rag/indexing/page/#vector-storage)). At search time, tune parameters to balance recall and latency.

```python
EFSEARCH_HIGH_RECALL = 50  # Default 16; increase for better recall at higher latency

index = faiss.read_index("index.faiss")
index.hnsw.efSearch = EFSEARCH_HIGH_RECALL
```

Set efSearch=100 when high stage 1 recall is critical (missing the correct chunk means it won't be reranked). Set efSearch=20 when latency is tight and you're reranking 100+ candidates to compensate.

---

## Search-time filtering & boosting

Filtering and boosting happen during search execution — they are not query rewrites but constraints and score adjustments applied to the retrieval process itself.

### Metadata filtering

Filter by document metadata before retrieval to reduce search space and improve precision.

```python
COLLECTION_NAME = "documents"
STAGE1_K = 50

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_embedding.tolist(),
    query_filter=Filter(must=[
        FieldCondition(key="category", match=MatchValue(value="healthcare")),
        FieldCondition(key="created_at", range=DatetimeRange(gte="2024-01-01T00:00:00Z")),
    ]),
    limit=STAGE1_K,
)
```

Helps when users specify constraints or for time-sensitive queries. Hurts when filters are too restrictive or metadata is incomplete.

### Hard security filtering (RBAC)

Hard security filtering physically blocks unauthorized chunks from retrieval results. This is distinct from score boosting — unauthorized users never see restricted content, period. Critical for enterprise RAG handling sensitive data.

```python
COLLECTION_NAME = "documents"
JWT_ALGORITHM = "HS256"


def secure_retrieval(
    query: str,
    user_token: str,
    encoder: SentenceTransformer,
    client: QdrantClient,
    secret_key: str,
    k: int = 10,
) -> list[dict[str, str | float]]:
    """Retrieves chunks filtered by JWT-derived RBAC roles."""
    if not query.strip():
        raise ValueError("query must be non-empty")

    payload = jwt.decode(user_token, secret_key, algorithms=[JWT_ALGORITHM])
    user_roles: list[str] = payload.get("roles", [])

    rbac_filter = Filter(must=[
        FieldCondition(key="roles", match=MatchAny(any=user_roles))
    ])

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(query).tolist(),
        query_filter=rbac_filter,
        limit=k,
    )
    return [{"text": hit.payload["text"], "score": hit.score} for hit in results]
```

**You must implement RBAC** for multi-tenant SaaS, regulated industries (HIPAA, GDPR, SOX, PCI-DSS), sensitive data types (PII, PHI, financial, legal), and enterprise contracts requiring SOC 2 or ISO 27001. **Soft boosting is sufficient** for single-tenant systems with public data only and <10 trusted users.

**Business value**:

| Item | Relative cost |
|------|---------------|
| Implementation cost | Moderate (identity integration, metadata audit, monitoring) |
| Annual operational cost | Low (ongoing sync, monitoring, audits) |
| Value protected: breach prevention | High (average breach costs are multiples of implementation cost) |
| Value protected: compliance | Very high (HIPAA/GDPR penalties can reach millions) |
| Payback period | Typically under 6 months for regulated industries |

**Concrete scenarios**:
- **Healthcare (HIPAA)**: Cardiologist blocked from psychiatry records. Civil penalties scale by violation tier; knowing violations can trigger criminal prosecution.
- **Multi-tenant SaaS**: Customer A must never see Customer B's data. Leakage = 100% churn + lawsuit.
- **Legal**: Attorney on Case A blocked from Case B discovery documents (conflict walls). Violation = malpractice + disbarment.
- **HR (GDPR)**: US HR admin blocked from EU employee compensation data. Violation = up to 4% global revenue.
- **Finance (SOX)**: Junior analyst blocked from internal deal flow and M&A targets. Failed audit = loss of enterprise customers.

**Hierarchical permissions**:

```python
# Department-level
metadata = {"roles": ["legal"], "departments": ["legal"]}
# Team-level
metadata = {"roles": ["legal", "compliance"], "team": "privacy-team"}
# Individual-level
metadata = {"roles": ["legal"], "owner_id": "user_456"}
```

**Audit trail implementation**:

```python
from datetime import datetime, UTC

COLLECTION_NAME = "documents"
JWT_ALGORITHM = "HS256"


def secure_retrieval_with_audit(
    query: str,
    user_token: str,
    encoder: SentenceTransformer,
    client: QdrantClient,
    secret_key: str,
    k: int = 10,
) -> list:
    """Retrieves RBAC-filtered chunks and logs an audit trail to SIEM."""
    if not query.strip():
        raise ValueError("query must be non-empty")

    payload = jwt.decode(user_token, secret_key, algorithms=[JWT_ALGORITHM])
    user_id: str = payload["user_id"]
    user_roles: list[str] = payload.get("roles", [])

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(query).tolist(),
        query_filter=Filter(must=[FieldCondition(key="roles", match=MatchAny(any=user_roles))]),
        limit=k,
    )

    audit_log = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user_id": user_id, "user_roles": user_roles,
        "query": query, "chunks_returned": len(results),
    }
    log_to_siem(audit_log)
    return results
```

**Security monitoring alerts**:
- Unusual access volume (10x daily average — possible exfiltration)
- Repeated failed access (5+ blocked queries/hour — unauthorized access attempt)
- Permission drift (IdP roles changed but vector DB metadata not updated in 24h)
- High blocking rate (80%+ consistently — user needs additional permissions)

**Compliance evidence requirements**:
- **SOC 2 Type II**: Access control enforcement metrics over an observation period (typically 12 months, minimum 3-6 months for first audits)
- **HIPAA § 164.312(b)**: Audit controls — hardware, software, and procedural mechanisms to record and examine access to ePHI
- **GDPR Article 30**: Records of processing activities including a general description of technical and organisational security measures
- **ISO 27001 A.5.15** (2022; formerly A.9.1): Access control policy based on business and security requirements + implementation evidence

**Technical trade-offs**: +5-15ms latency, +5-10% storage for RBAC metadata. Always validate JWT signatures and expiration. Sync permissions regularly between IdP and vector DB or use short-lived tokens (1-hour TTL).

**Pitfalls**: Missing RBAC metadata at indexing time forces reindexing. Trusting unvalidated JWT tokens is a critical vulnerability. Permission drift between IdP and vector DB causes stale access — sync regularly.

**Cross-references**: See [indexing metadata strategy]({{ site.baseurl }}/docs/genai/rag/indexing/page/#metadata-strategy) for structuring RBAC fields at indexing time.

---

### Score boosting

Adjust retrieval scores based on metadata to promote authoritative, recent, or contextually relevant documents.

```python
from datetime import datetime, UTC

EPSILON = 1e-8
RECENCY_BOOST = 0.2
DAYS_PER_YEAR = 365
AUTHORITY_BOOST = 1.1
DEPARTMENT_BOOST = 1.15
TRUSTED_AUTHORS = frozenset({"legal-team", "compliance-team"})


def metadata_boosted_search(
    query: str,
    documents: list[dict],
    doc_embeddings: np.ndarray,
    encoder: SentenceTransformer,
    user_department: str,
    k: int = 10,
) -> list[dict]:
    """Applies recency, authority, and department boosts to base retrieval scores."""
    base_scores = np.dot(doc_embeddings, encoder.encode(query))
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + EPSILON)

    for i, doc in enumerate(documents):
        age_days = (datetime.now(UTC) - datetime.fromisoformat(doc["created_at"])).days
        base_scores[i] *= (1 + RECENCY_BOOST * np.exp(-age_days / DAYS_PER_YEAR))
        if doc.get("author") in TRUSTED_AUTHORS:
            base_scores[i] *= AUTHORITY_BOOST
        if doc.get("category") == user_department:
            base_scores[i] *= DEPARTMENT_BOOST

    return [documents[i] for i in np.argsort(base_scores)[::-1][:k]]
```

Keep boosts modest (10-30%). A recent irrelevant doc shouldn't beat an older highly relevant doc.

### Positional boosting

Introduction and conclusion sections often contain key concepts and summaries. Boost chunks based on position to improve precision for conceptual queries.

**Lost in the middle**: LLMs exhibit U-shaped performance — >20% drop when relevant information is in the middle of long contexts. This affects both which chunks to retrieve and how to order them for the LLM.

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Places most relevant chunks at start and end to mitigate lost-in-the-middle effect."""
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back
```

```python
EPSILON = 1e-8
FIRST_CHUNK_BOOST = 1.2
LAST_CHUNK_BOOST = 1.15
HEADER_BOOST = 1.1
MIDDLE_PENALTY_FACTOR = 0.15
LONG_DOC_THRESHOLD = 10

SECTION_WEIGHTS: dict[str, dict[str, float]] = {
    "research_paper": {"abstract": 1.15, "methodology": 1.25, "results": 1.2, "references": 0.8},
    "technical_doc": {"implementation": 1.3, "api_reference": 1.2, "troubleshooting": 1.15},
    "legal_doc": {"definitions": 1.3, "requirements": 1.25, "obligations": 1.25, "preamble": 0.9},
}


def positional_boosted_search(
    query: str,
    documents: list[dict],
    doc_embeddings: np.ndarray,
    encoder: SentenceTransformer,
    k: int = 10,
) -> list[dict]:
    """Boosts retrieval scores based on chunk position within the source document."""
    base_scores = np.dot(doc_embeddings, encoder.encode(query))
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + EPSILON)
    boosted_scores = base_scores.copy()

    for i, doc in enumerate(documents):
        chunk_position = doc.get("chunk_index", 0)
        total_chunks = doc.get("total_chunks", 1)

        if chunk_position == 0:
            boosted_scores[i] *= FIRST_CHUNK_BOOST
        elif chunk_position == total_chunks - 1:
            boosted_scores[i] *= LAST_CHUNK_BOOST
        if doc.get("is_section_header", False):
            boosted_scores[i] *= HEADER_BOOST
        if total_chunks > LONG_DOC_THRESHOLD:
            normalized_position = chunk_position / (total_chunks - 1)
            position_weight = 1 - MIDDLE_PENALTY_FACTOR * (1 - 4 * (normalized_position - 0.5) ** 2)
            boosted_scores[i] *= position_weight

    return [documents[i] for i in np.argsort(boosted_scores)[::-1][:k]]
```

Helps for structured documents (papers, reports, technical docs). Hurts for unstructured content (chat logs, transcripts) or queries seeking specific details in middle sections. Store positional metadata during indexing. Combine with reranking: apply positional reordering after cross-encoder, not before — let the reranker score on pure relevance first.

---

## Query optimization

Queries are not always well-formed. Users ask "it", "more on that", or complex multi-part questions. Query optimization rewrites queries before retrieval.

### Query expansion

Add synonyms or related terms to increase recall. Use T5 or similar to generate paraphrases, concatenate with original query for retrieval.

```python
MAX_PARAPHRASE_LENGTH = 50
NUM_PARAPHRASES = 3
NUM_BEAMS = 5


def expand_query(query: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration) -> str:
    """Generates paraphrases and concatenates them with the original query for broader recall."""
    input_ids = tokenizer(f"paraphrase: {query}", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, max_length=MAX_PARAPHRASE_LENGTH,
        num_return_sequences=NUM_PARAPHRASES, num_beams=NUM_BEAMS,
    )
    paraphrases = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return query + " " + " ".join(paraphrases)
```

Helps for short/ambiguous queries and domain abbreviations ("ML" → "machine learning"). Hurts for long, specific queries where expansions add noise.

### Conversational query rewriting

In multi-turn conversations, follow-up queries like "What are its limitations?" lack context. Rewriting reformulates them into standalone questions using chat history.

```python
HISTORY_WINDOW = 5


def rewrite_conversational_query(query: str, chat_history: list[dict[str, str]], llm: LLM) -> str:
    """Rewrites a follow-up query into a standalone question using recent chat history."""
    recent_history = chat_history[-HISTORY_WINDOW:]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
    prompt = f"""Rewrite the user's latest query into a standalone question.

Conversation history:
{history_text}

Latest query: {query}

Standalone query:"""
    return llm.invoke(prompt).content.strip()
```

**Detecting when rewriting is needed**: Check for pronouns and contextual references (it, its, this, that, also, instead). Skip rewriting for standalone queries to avoid unnecessary LLM calls. Simple regex works as a first pass but produces false positives — "What is the `this` keyword?" or "What is the difference between X and Y?" would trigger unnecessarily. Treat it as a cheap gate, not a precise classifier.

**Latency optimization**: Use a smaller, faster model for rewriting (an order of magnitude faster than large models), cache rewritten queries by (query + history hash), or fine-tune a small seq2seq model (e.g., T5-small) for significantly faster rewrites with competitive quality on narrow rewriting tasks.

```python
import hashlib
import json
import re

CONTEXTUAL_REFERENCE_PATTERNS = [
    r"\b(it|its|this|that|these|those|they|them|their)\b",
    r"\b(also|too|as well|additionally|furthermore)\b",
    r"\b(instead|however)\b",
]


def needs_rewriting(query: str) -> bool:
    """Cheap heuristic gate: detects pronouns and contextual references."""
    return any(re.search(p, query, re.IGNORECASE) for p in CONTEXTUAL_REFERENCE_PATTERNS)


def _chat_history_hash(chat_history: list[dict[str, str]]) -> str:
    """Deterministic hash of chat history for cache keying."""
    return hashlib.md5(
        json.dumps(chat_history, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()


def rewrite_with_cache(
    query: str,
    chat_history: list[dict[str, str]],
    llm: LLM,
    cache: dict[str, str],
) -> str:
    """Returns cached rewrite if available; otherwise rewrites and caches."""
    cache_key = f"{query}:{_chat_history_hash(chat_history)}"
    if cache_key in cache:
        return cache[cache_key]
    rewritten = rewrite_conversational_query(query, chat_history, llm)
    cache[cache_key] = rewritten
    return rewritten
```

**Fine-tuned rewriting model** (orders of magnitude faster than LLM-based rewriting):

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

REWRITER_MODEL = "your-org/t5-query-rewriter"
HISTORY_WINDOW = 3
MAX_REWRITE_LENGTH = 100


class FastQueryRewriter:
    """Fine-tuned T5 model for sub-millisecond conversational query rewriting."""

    def __init__(self, model_name: str = REWRITER_MODEL) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def rewrite(self, query: str, chat_history: list[dict[str, str]]) -> str:
        """Rewrites a follow-up query into a standalone question using recent context."""
        history_text = " ".join([msg["content"] for msg in chat_history[-HISTORY_WINDOW:]])
        input_ids = self.tokenizer(
            f"rewrite query with context: {history_text} [SEP] {query}", return_tensors="pt"
        ).input_ids
        outputs = self.model.generate(input_ids, max_length=MAX_REWRITE_LENGTH)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Critical for** chatbot interfaces, multi-turn Q&A, and voice assistants. **Less important for** single-turn search, keyword search, and FAQ systems.

### Query decomposition

Break multi-part questions into sub-queries, retrieve separately, merge and rerank with the original query.

```python
SUB_QUERY_K = 20
FINAL_K = 5


def decompose_query(query: str, llm: LLM) -> list[str]:
    """Breaks a multi-part question into 2-4 independent sub-queries via LLM."""
    prompt = f"Break this into 2-4 simple sub-questions:\n\nQuestion: {query}\n\nSub-questions:"
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]


# "Compare HIPAA and GDPR for healthcare data" →
# 1. What are HIPAA's privacy requirements?
# 2. What are GDPR's privacy requirements?
# 3. How do they differ for healthcare data?

all_results: dict[int, float] = {}
for sub_q in decompose_query(query, llm):
    for doc_id, score in retrieve(sub_q, documents, k=SUB_QUERY_K):
        all_results[doc_id] = max(all_results.get(doc_id, 0), score)
top_chunks = rerank(query, list(all_results.keys()), k=FINAL_K)
```

Helps for multi-hop questions ("compare X and Y"). Hurts for simple queries where decomposition adds significant overhead (LLM call for decomposition + multiple retrieval rounds).

### Self-querying (natural language to metadata filters)

When a user asks "Show me healthcare policies from 2023", running semantic search on the full text is inefficient — "2023" will pollute the embedding. **Self-querying** uses an LLM (or a smaller model) to extract structured metadata filters from the query text, then passes only the semantic portion to the vector search.

```
User query: "healthcare policies from 2023"
    ↓ LLM extraction
Semantic query: "healthcare policies"
Metadata filters: {year: >= 2023}
    ↓
Vector search with pre-filters applied
```

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(name="category", description="Document category", type="string"),
    AttributeInfo(name="year", description="Year published", type="integer"),
    AttributeInfo(name="author", description="Author name", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Company policy documents",
    metadata_field_info=metadata_field_info,
)

# "healthcare policies from 2023" →
#   query="healthcare policies", filter=gte("year",2023)
```

The LLM receives a structured prompt describing available metadata fields (names, types, descriptions) and outputs a structured query with a semantic text component and filter expressions. LangChain's `SelfQueryRetriever` and LlamaIndex's `VectorIndexAutoRetriever` both implement this pattern.

**Lightweight alternatives**: For predictable metadata patterns, skip the LLM call entirely. Use regex or NER models to extract dates, categories, and named entities — orders of magnitude faster than an LLM call. Reserve LLM-based extraction for open-ended metadata schemas or complex filter logic (nested AND/OR conditions).

**When self-querying helps**: Queries with explicit constraints (dates, categories, authors, document types) that map to indexed metadata fields. **When to skip**: Purely semantic queries without metadata constraints, or when metadata fields are sparse/unreliable.

### Advanced query techniques

#### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, embed it instead of the raw query. Documents are written as answers — matching answer-to-answer is more effective than matching question-to-answer.

```python
def hyde_retrieval(
    query: str, documents: list[str], encoder: SentenceTransformer, llm: LLM, k: int = 5,
) -> list[str]:
    """Generates a hypothetical answer and uses its embedding for retrieval."""
    hypothetical_answer = llm.invoke(
        f'Given the question: "{query}"\nWrite a detailed, factual answer.'
    ).content
    hyde_embedding = encoder.encode(hypothetical_answer)
    similarities = np.dot(encoder.encode(documents), hyde_embedding)
    return [documents[i] for i in np.argsort(similarities)[::-1][:k]]
```

Helps for complex queries with large semantic gaps between query and document vocabulary. Hurts when the hypothetical answer is wrong (leads retrieval astray) or for simple queries. Adds a full LLM generation round-trip. Optimization: average 3-5 hypothetical answer embeddings for robustness.

#### Multi-Query

Generate multiple query variations from different perspectives, retrieve for all, merge with RRF.

```python
NUM_VARIATIONS = 3
STAGE1_K = 50
RRF_K = 60


def multi_query_retrieval(
    query: str, documents: list[str], encoder: SentenceTransformer, llm: LLM, k: int = 5,
) -> list[str]:
    """Generates query variations, retrieves for each, and merges via RRF."""
    variations = llm.invoke(
        f"Generate {NUM_VARIATIONS} different versions of: {query}"
    ).content.split("\n")[:NUM_VARIATIONS]

    doc_embeddings = encoder.encode(documents)
    scores: dict[int, float] = {}
    for q in [query] + variations:
        sims = np.dot(doc_embeddings, encoder.encode(q))
        ranked = np.argsort(sims)[::-1][:STAGE1_K]
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

    return [documents[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]
```

Helps for ambiguous queries and domain-specific terminology variation. Adds one LLM call + multiple retrieval rounds.

#### Step-back prompting

Generate a more general query to retrieve foundational context alongside specific details. Originally proposed for reasoning tasks (Zheng et al., 2023, Google DeepMind), adapted for retrieval by practitioners and frameworks like LangChain.

```python
GENERAL_FRACTION = 1 / 3


def step_back_retrieval(query: str, documents: list[str], llm: LLM, k: int = 10) -> list:
    """Retrieves with both the original and a generalized query, then reranks the union."""
    step_back_query = llm.invoke(
        f'Generate a more general question for: "{query}"'
    ).content.strip()

    specific = retrieve(query, documents, k=k)
    general = retrieve(step_back_query, documents, k=k)
    n_general = int(k * GENERAL_FRACTION)
    n_specific = k - n_general
    seen: set[int] = set()
    combined: list[int] = []
    for doc_id, score in specific[:n_specific] + general[:n_general]:
        if doc_id not in seen:
            seen.add(doc_id)
            combined.append(doc_id)
    return rerank(query, combined, k=k)
```

Helps for complex domain questions requiring background context. Hurts for simple factual queries. Adds one extra LLM call for the step-back generation.

---

## Result optimization

Query optimization transforms the query before retrieval. Result optimization processes retrieved chunks to improve diversity, reduce redundancy, or expand context.

### MMR (Maximal Marginal Relevance)

Standard retrieval can return 5 chunks discussing the same narrow aspect. MMR balances relevance and diversity: `MMR = lambda * sim(query, chunk) - (1-lambda) * max(sim(chunk, selected_chunks))`.

```python
def mmr_rerank(
    query: str,
    candidates: list[str],
    encoder: SentenceTransformer,
    k: int = 5,
    lambda_param: float = 0.5,
) -> list[str]:
    """Selects k candidates balancing relevance to query and diversity among selections."""
    if k > len(candidates):
        raise ValueError(f"k={k} exceeds candidate count {len(candidates)}")

    query_emb = encoder.encode(query)
    cand_embs = encoder.encode(candidates)
    relevance = np.dot(cand_embs, query_emb)

    selected = [int(np.argmax(relevance))]
    sel_embs = [cand_embs[selected[0]]]

    while len(selected) < k:
        remaining = [i for i in range(len(candidates)) if i not in selected]
        mmr_scores = []
        for i in remaining:
            diversity_penalty = max(np.dot(cand_embs[i], e) for e in sel_embs)
            mmr_scores.append((i, lambda_param * relevance[i] - (1 - lambda_param) * diversity_penalty))
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        sel_embs.append(cand_embs[best])

    return [candidates[i] for i in selected]
```

**Tuning lambda**: 0.7-0.9 favor relevance (precision search), 0.5 balanced default, 0.3-0.5 favor diversity (exploratory/multi-faceted queries). Apply MMR after reranking (100 → 50 with cross-encoder, then MMR 50 → 5). Computational cost scales O(k × n) where n is the candidate pool size.

### Contextual compression

Not every sentence in a chunk is relevant. Compression filters chunks to keep only query-relevant content.

**Embedding-based filtering** (fast, sub-second on GPU): re-embed sentences, keep those above similarity threshold.

```python
def embedding_compress(
    query: str, chunks: list[str], encoder: SentenceTransformer, threshold: float = 0.5,
) -> list[str]:
    """Keeps only sentences whose embedding similarity to the query exceeds threshold."""
    query_emb = encoder.encode(query)
    compressed = []
    for chunk in chunks:
        sentences = chunk.split(". ")
        relevant = [
            s for s, sim in zip(sentences, np.dot(encoder.encode(sentences), query_emb))
            if sim >= threshold
        ]
        if relevant:
            compressed.append(". ".join(relevant))
    return compressed
```

**LLM-based extraction** (high quality, one LLM call per chunk): prompt LLM to extract relevant sentences verbatim. Typically ~2-5x compression but expensive at scale.

```python
EXTRACTION_PROMPT = """Extract ONLY sentences relevant to the query. Return empty if none.
Query: {query}
Chunk: {chunk}
Relevant sentences:"""


def llm_compress(query: str, chunks: list[str], llm: LLM) -> list[str]:
    """Extracts query-relevant sentences from each chunk via LLM."""
    return [
        r for chunk in chunks
        if (r := llm.invoke(EXTRACTION_PROMPT.format(query=query, chunk=chunk)).content.strip())
    ]


def pipeline_compress(
    query: str,
    chunks: list[str],
    encoder: SentenceTransformer,
    redundancy_threshold: float = 0.85,
    relevance_threshold: float = 0.5,
) -> list[str]:
    """Deduplicates sentences by embedding similarity, then filters by query relevance."""
    query_emb = encoder.encode(query)
    all_sentences = [s.strip() for chunk in chunks for s in chunk.split(". ") if s.strip()]
    sent_embs = encoder.encode(all_sentences)

    unique_sents: list[str] = []
    unique_embs: list[np.ndarray] = []
    for sent, emb in zip(all_sentences, sent_embs):
        if not unique_embs or np.max(np.dot(unique_embs, emb)) < redundancy_threshold:
            unique_sents.append(sent)
            unique_embs.append(emb)

    scores = np.dot(unique_embs, query_emb)
    return [s for s, score in zip(unique_sents, scores) if score >= relevance_threshold]
```

**Threshold tuning**: 0.3 permissive (retains most content), 0.5 balanced default, 0.7 aggressive (risks removing important context). Exact compression ratios vary significantly by model, domain, and query distribution. Prefer applying after reranking to avoid losing relevant content before scoring. Start with embedding-based; use LLM-based only when precision is critical.

### Parent-child retrieval

Small chunks match queries precisely but lack context. Large chunks provide context but match poorly. Parent-child retrieval solves this: retrieve based on small chunks, return large chunks to the LLM.

**Sentence window retrieval**: Embed individual sentences. When retrieved, expand to +/-N surrounding sentences (default N=3, yielding 7 sentences). Simple, works for linear documents.

**Auto-merging retrieval**: Build a hierarchical tree (parent 2048 tokens → intermediate 512 → leaf 128). Retrieve leaf nodes; if 50%+ of a parent's children are retrieved, return the parent instead. Better for structured documents.

| Aspect | Sentence Window | Auto-Merging |
|--------|-----------------|--------------|
| **Expansion** | Fixed (always expands) | Selective (threshold-based) |
| **Context type** | Local (adjacent) | Hierarchical (structured) |
| **Implementation** | Simple | Complex |
| **Token efficiency** | Lower | Higher |
| **Best for** | Linear documents | Structured documents |

**Sentence window:**

```python
class SentenceWindowRetriever:
    """Embeds individual sentences and expands to surrounding context at search time."""

    def __init__(self, documents: list[str], encoder: SentenceTransformer, window_size: int = 3) -> None:
        self.window_size = window_size
        self.encoder = encoder
        self.sentences: list[str] = []
        self.metadata: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            doc_sents = [s.strip() + "." for s in doc.split(".") if s.strip()]
            for sent_idx, sent in enumerate(doc_sents):
                self.sentences.append(sent)
                self.metadata.append({"doc_idx": doc_idx, "sent_idx": sent_idx, "doc_sents": doc_sents})
        self.embeddings = self.encoder.encode(self.sentences)

    def search(self, query: str, k: int = 5) -> list[dict[str, str]]:
        """Returns k sentence windows most similar to the query."""
        sims = np.dot(self.embeddings, self.encoder.encode(query))
        windows: list[dict[str, str]] = []
        for idx in np.argsort(sims)[::-1][:k]:
            m = self.metadata[idx]
            start = max(0, m["sent_idx"] - self.window_size)
            end = min(len(m["doc_sents"]), m["sent_idx"] + self.window_size + 1)
            windows.append({"window": " ".join(m["doc_sents"][start:end]), "target": self.sentences[idx]})
        return windows
```

**Auto-merging:**

```python
from pydantic import BaseModel

LEAF_LEVEL = 2


class HierarchicalNode(BaseModel):
    """A node in a parent-intermediate-leaf chunking hierarchy."""

    chunk_id: str
    text: str
    level: int
    parent_id: str | None
    children_ids: list[str]


def auto_merge_retrieve(
    query: str,
    nodes: dict[str, HierarchicalNode],
    encoder: SentenceTransformer,
    k: int = 12,
    merge_threshold: float = 0.5,
) -> list[str]:
    """Retrieves leaf nodes and merges into parent text when enough siblings match."""
    leaf_nodes = [n for n in nodes.values() if n.level == LEAF_LEVEL]
    leaf_embs = encoder.encode([n.text for n in leaf_nodes])
    top_leaves = [leaf_nodes[i] for i in np.argsort(np.dot(leaf_embs, encoder.encode(query)))[::-1][:k]]

    parent_to_children: dict[str | None, list[str]] = {}
    for leaf in top_leaves:
        parent_to_children.setdefault(leaf.parent_id, []).append(leaf.chunk_id)

    final: list[str] = []
    for parent_id, child_ids in parent_to_children.items():
        parent = nodes[parent_id]
        if len(child_ids) / len(parent.children_ids) >= merge_threshold:
            final.append(parent.text)
        else:
            final.extend(nodes[cid].text for cid in child_ids)
    return final
```

**Tuning**: Window size 3 is the recommended default (LlamaIndex default; benchmarks show groundedness can drop at larger sizes as context overwhelms the LLM). Merge threshold 0.5 is the default (LlamaIndex); 0.3-0.4 aggressive, 0.6-0.7 conservative. Always apply reranking after expansion to filter the best expanded contexts.

### Document-level vs chunk-level retrieval

Sometimes return entire documents instead of chunks. Short documents (<4K tokens), single-purpose content, and queries requiring narrative coherence benefit from document-level retrieval.

**Hybrid approach**: Index small chunks (400 tokens) for precise matching, retrieve and return larger parent chunks (2000 tokens) or full documents to the LLM.

```python
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP,
    ),
    parent_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP,
    ),
)
```

| Aspect | Chunk-Level | Document-Level | Parent Document |
|--------|-------------|----------------|-----------------|
| **Precision** | High | Low | High (child chunks) |
| **Context** | Low | High | High (parent chunks) |
| **Token Cost** | Low | High | Medium-High |

**Rough heuristic**: very short documents that fit in context may not need chunking; mid-length documents benefit from parent document retrieval; long documents need standard chunking. Route summarization queries to document-level, specific questions to chunk-level.

### Context enrichment

Pass metadata to the LLM for attribution. Include source, page, date, author — enough for users to verify. Prompt the LLM to cite sources explicitly ([1], [2]).

```python
COLLECTION_NAME = "documents"
CITATION_PROMPT = "Answer using only these sources. Cite by number."


def retrieve_with_metadata(
    query: str, encoder: SentenceTransformer, client: QdrantClient, k: int = 5,
) -> str:
    """Retrieves chunks with source metadata and formats a citation-ready LLM prompt."""
    results = client.search(
        collection_name=COLLECTION_NAME, query_vector=encoder.encode(query).tolist(), limit=k,
    )
    context = []
    for i, r in enumerate(results):
        m = r.payload
        context.append(
            f"[Source {i+1}] {m.get('source', '?')} "
            f"(p.{m.get('page_number', '?')}, {m.get('created_at', '?')})\n{r.payload['text']}"
        )
    separator = "\n\n"
    return f"{CITATION_PROMPT}\n\n{separator.join(context)}\n\nQuestion: {query}"
```

---

## Semantic caching

Caching retrieval results by query similarity rather than exact string match is a major architectural optimization. **Semantic caching** compares new queries against cached query embeddings (cosine >= 0.90-0.95), handling paraphrased queries and achieving significantly higher hit rates than exact-match caching.

Semantic caching is typically the first routing intercept — check the cache before any retrieval work.

```python
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600


class SemanticCache:
    """Caches retrieval results keyed by query embedding similarity."""

    def __init__(
        self, redis_client: Redis, encoder: SentenceTransformer,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD, ttl: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.redis = redis_client
        self.encoder = encoder
        self.threshold = threshold
        self.ttl = ttl

    def get(self, query: str) -> list | None:
        """Returns cached results if a similar query exists, else None."""
        ...

    def set(self, query: str, results: list) -> None:
        """Stores query embedding and results with TTL."""
        ...


def retrieve_with_cache(
    query: str, cache: SemanticCache, retriever: HybridRetriever, **kwargs,
) -> dict[str, list | bool]:
    """Checks semantic cache before falling back to live retrieval."""
    cached = cache.get(query)
    if cached:
        return {"results": cached, "from_cache": True}
    response = retriever.search(query, **kwargs)
    cache.set(query, response["results"])
    return {"results": response["results"], "from_cache": False}
```

```python
import hashlib
import json

import numpy as np
from redis import Redis

DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600
QUERY_ID_LENGTH = 16
CACHE_INDEX_KEY = "cache:query_index"


class SemanticCache:
    """Redis-backed semantic cache with cosine similarity matching."""

    def __init__(
        self, redis_client: Redis, encoder: SentenceTransformer,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.redis = redis_client
        self.encoder = encoder
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

    def _query_id(self, embedding: np.ndarray) -> str:
        """Derives a short deterministic ID from an embedding vector."""
        return hashlib.sha256(embedding.astype(np.float32).tobytes()).hexdigest()[:QUERY_ID_LENGTH]

    def get(self, query: str) -> list | None:
        """Returns cached results for the most similar query, or None on miss."""
        query_emb = self.encoder.encode(query)
        cached_ids = self.redis.smembers(CACHE_INDEX_KEY)
        if not cached_ids:
            return None

        # NOTE: This linear scan is O(n) per lookup — pedagogical only.
        # Production systems should use Redis Stack's vector search (HNSW indexing)
        # or a dedicated vector store for sub-linear lookup.
        max_sim, best_id = -1.0, None
        for qid in cached_ids:
            qid = qid.decode("utf-8")
            cached_emb = json.loads(self.redis.get(f"cache:query:{qid}") or "null")
            if cached_emb is None:
                self.redis.srem(CACHE_INDEX_KEY, qid)
                continue
            sim = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if sim > max_sim:
                max_sim, best_id = sim, qid

        if max_sim >= self.threshold and best_id:
            results = self.redis.get(f"cache:results:{best_id}")
            return json.loads(results) if results else None
        return None

    def set(self, query: str, results: list) -> None:
        """Stores query embedding and results with TTL."""
        emb = self.encoder.encode(query)
        qid = self._query_id(emb)
        # NOTE: These three calls are non-atomic — production code should use
        # a Redis pipeline or transaction to avoid partial state on failure.
        self.redis.setex(f"cache:query:{qid}", self.ttl, json.dumps(emb.tolist()))
        self.redis.setex(f"cache:results:{qid}", self.ttl, json.dumps(results))
        self.redis.sadd(CACHE_INDEX_KEY, qid)
```

**Threshold tuning**: 0.85-0.90 (higher hit rate, lower precision — narrow/repetitive corpora), 0.90-0.95 (balanced — recommended default), 0.95-0.99 (low hit rate, very high precision — compliance-critical). The right threshold depends on corpus breadth: narrow FAQ corpora tolerate lower thresholds, while broad corpora need higher thresholds to avoid wrong answers. Exact hit-rate and precision numbers vary significantly by query distribution and domain. For false positive management: seed with verified pairs, sample borderline hits for review, optionally validate with cross-encoder.

**Cache invalidation**: Use short TTLs for frequently changing content, longer TTLs for stable content, and longest for reference material. Add event-driven invalidation for critical accuracy. Alternatives: GPTCache, LangChain semantic cache (`RedisSemanticCache`), Redis Stack.

---

## Workflow: building a retrieval pipeline

Retrieval is a pipeline: query processing → stage 1 retrieval → stage 2 reranking → result validation. Start simple, add complexity as needed.

**Simple dense retrieval** (prototype, 20 lines):

```python
ENCODER_MODEL = "all-mpnet-base-v2"

model = SentenceTransformer(ENCODER_MODEL)
index = faiss.read_index("index.faiss")


def search(query: str, k: int = 5) -> list[dict[str, str | float]]:
    """Retrieves k nearest chunks from a FAISS index."""
    query_embedding = model.encode(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_embedding, k=k)
    return [{"chunk": chunks[i], "distance": float(distances[0][j])} for j, i in enumerate(indices[0])]
```

**Two-stage pipeline** (production, significant MRR improvement):

```python
def two_stage_search(query: str, stage1_k: int = 50, final_k: int = 5) -> list[str]:
    """Fast ANN retrieval followed by cross-encoder reranking."""
    query_embedding = retriever.encode(query).reshape(1, -1).astype("float32")
    _, indices = index.search(query_embedding, k=stage1_k)
    stage1_chunks = [chunks[i] for i in indices[0]]

    rerank_scores = reranker.predict([[query, c] for c in stage1_chunks])
    return [stage1_chunks[i] for i in np.argsort(rerank_scores)[::-1][:final_k]]
```

**Hybrid with reranking** (production, handles diverse query types):

```python
ENCODER_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
EPSILON = 1e-8


class HybridRetriever:
    """Two-stage retriever: hybrid dense+sparse search followed by cross-encoder reranking."""

    def __init__(self, index_path: str, chunks_path: str) -> None:
        self.retriever = SentenceTransformer(ENCODER_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.index = faiss.read_index(index_path)
        self.chunks = np.load(chunks_path, allow_pickle=True)
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
        self, query: str, stage1_k: int = 100, final_k: int = 5, alpha: float = 0.5,
    ) -> list[dict[str, str | float]]:
        """Runs hybrid retrieval (dense + BM25) then cross-encoder reranking."""
        query_embedding = self.retriever.encode(query)
        dense_distances, dense_indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), k=stage1_k)
        # FAISS returns squared L2 distances; sqrt converts to true L2 before normalization
        dense_scores = 1 / (1 + np.sqrt(dense_distances[0]))

        sparse_scores = self.bm25.get_scores(query.lower().split())

        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + EPSILON)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + EPSILON)

        hybrid_scores = np.zeros(len(self.chunks))
        for i, idx in enumerate(dense_indices[0]):
            hybrid_scores[idx] = alpha * dense_norm[i]
        hybrid_scores += (1 - alpha) * sparse_norm

        top_indices = np.argsort(hybrid_scores)[::-1][:stage1_k]
        stage1_chunks = [self.chunks[i] for i in top_indices]

        rerank_scores = self.reranker.predict([[query, c] for c in stage1_chunks])
        reranked = np.argsort(rerank_scores)[::-1][:final_k]
        return [{"chunk": stage1_chunks[i], "rerank_score": float(rerank_scores[i])} for i in reranked]
```

**Production pipeline with monitoring:**

```python
import logging
import time
from collections import deque

from pydantic import BaseModel

logger = logging.getLogger(__name__)

METRICS_WINDOW = 1000
SUMMARY_WINDOW = 100
MS_PER_SECOND = 1000


class RetrievalMetrics(BaseModel):
    """Latency telemetry for a single retrieval request."""

    query: str
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms: float


class ProductionRetriever:
    """Wraps HybridRetriever with query caching, latency tracking, and logging."""

    def __init__(self, index_path: str, chunks_path: str) -> None:
        self._retriever = HybridRetriever(index_path, chunks_path)
        self._metrics: deque[RetrievalMetrics] = deque(maxlen=METRICS_WINDOW)
        self._embed_cache: dict[str, np.ndarray] = {}

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._embed_cache:
            self._embed_cache[query] = self._retriever.retriever.encode(query)
        return self._embed_cache[query]

    def search(
        self, query: str, stage1_k: int = 100, final_k: int = 5, alpha: float = 0.5,
    ) -> dict[str, list | dict]:
        """Runs hybrid retrieval and records latency metrics."""
        start = time.perf_counter()

        s1_start = time.perf_counter()
        self._embed_query(query)
        results = self._retriever.search(query, stage1_k=stage1_k, final_k=final_k, alpha=alpha)
        s1_ms = (time.perf_counter() - s1_start) * MS_PER_SECOND

        total_ms = (time.perf_counter() - start) * MS_PER_SECOND
        self._metrics.append(RetrievalMetrics(
            query=query, stage1_latency_ms=s1_ms,
            stage2_latency_ms=0, total_latency_ms=total_ms))
        logger.info("Retrieved %d chunks in %.1fms", final_k, total_ms)
        return {"results": results, "metrics": {"total_latency_ms": total_ms}}

    def get_performance_summary(self) -> dict[str, float]:
        """Returns average and p95 latency over the most recent requests."""
        recent = list(self._metrics)[-SUMMARY_WINDOW:]
        latencies = [m.total_latency_ms for m in recent]
        return {
            "avg_total_latency_ms": float(np.mean(latencies)),
            "p95_total_latency_ms": float(np.percentile(latencies, 95)),
        }
```

---

## Agentic retrieval

The entire pipeline described above assumes a fixed DAG: query → rewrite → retrieve → rerank → generate. **Agentic retrieval** replaces this with an LLM agent that has retrieval as a tool — the agent decides when, how often, and with what query to search, and can iterate until it has enough context.

### Pipeline vs agentic RAG

| Aspect | Pipeline RAG | Agentic RAG |
|--------|-------------|-------------|
| **Architecture** | Fixed DAG (query → retrieve → generate) | Agent loop (reason → act → observe → repeat) |
| **Retrieval calls** | 1 (sometimes 2-3 with decomposition) | 1-10+ (agent decides) |
| **Latency** | Predictable (sub-second retrieval) | Variable (seconds to tens of seconds depending on iterations) |
| **Cost** | Low (1 LLM call for generation) | Higher (multiple LLM calls for reasoning) |
| **Best for** | Factual lookups, FAQ, single-hop questions | Multi-hop reasoning, research, complex analysis |
| **Predictability** | High (deterministic path) | Lower (agent may take unexpected paths) |

**When to use pipeline RAG**: Latency-sensitive applications, predictable query types, FAQ/support systems, cost-sensitive workloads. **When to use agentic RAG**: Complex multi-hop questions, research tasks requiring synthesis across many sources, queries where the first retrieval may not contain the full answer.

### Key agentic patterns

**Tool-calling agent**: Give the LLM a `search_docs` tool. It queries, reads results, realizes it has a partial answer, formulates a new query for the missing part, and searches again. Implemented in LangGraph, LlamaIndex agents, and most agent frameworks.

```python
MAX_AGENT_ITERATIONS = 5
RETRIEVAL_K = 5


def agentic_retrieve(
    question: str,
    retriever: HybridRetriever,
    llm: LLM,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> str:
    """Iteratively retrieves documents until the LLM has enough context to answer."""
    tools = [{"name": "search_docs", "fn": lambda q: retriever.search(q, k=RETRIEVAL_K)}]
    messages: list[dict[str, str]] = [{"role": "user", "content": question}]
    response = None
    for _ in range(max_iterations):
        response = llm.invoke(messages, tools=tools)
        messages.append(response.message)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            results = tools[0]["fn"](call.arguments["query"])
            messages.append({"role": "tool", "content": format_results(results)})
    return response.content if response else ""
```

**Self-RAG** (Asai et al., 2023; ICLR 2024 oral, top 1%): The model generates special reflection tokens — `[Retrieve]` (should I retrieve?), `[IsRel]` (is the retrieved passage relevant?), `[IsSup]` (is the generation supported?), `[IsUse]` (is the response useful?) — enabling adaptive retrieval. The model only retrieves when it determines its knowledge is insufficient, and self-evaluates the quality of retrieved context before generating.

**Corrective RAG (CRAG)** (Yan et al., 2024): Adds a lightweight retrieval evaluator (fine-tuned T5-large) that classifies retrieved documents as Correct, Incorrect, or Ambiguous. On Correct, it applies a decompose-then-recompose algorithm to strip noise. On Incorrect, it discards retrieval and triggers web search as a fallback. On Ambiguous, it combines refined retrieval with web results. CRAG is plug-and-play — it layers on top of any existing RAG pipeline.

**Frameworks**: LangGraph (graph-based agent orchestration with explicit state), LlamaIndex (AgentWorkflow with built-in retrieval tools), Haystack (lower framework overhead than LangChain), CrewAI (multi-agent collaboration), and AG2 (formerly AutoGen; community fork after Microsoft retired AutoGen in favor of Microsoft Agent Framework) all support agentic RAG patterns.

**Practical recommendation**: Use pipeline RAG by default and trigger an agentic loop only when failure signals are detected — low retrieval confidence, missing citations, contradictory evidence, or user follow-ups indicating the initial answer was insufficient. This keeps most queries efficient while providing a recovery path for complex cases.

---

## Beyond text: emerging retrieval paradigms

Standard RAG retrieval assumes text-in, text-out. Two emerging paradigms extend retrieval beyond this boundary.

### GraphRAG (Knowledge Graph Retrieval)

Vector similarity retrieval struggles with queries requiring synthesis across many documents ("What are the main themes in this dataset?" or multi-hop reasoning "How does X relate to Y through Z?"). **GraphRAG** builds a knowledge graph from documents — extracting entities and relationships — then uses graph traversal (community detection, path finding) alongside vector similarity.

Microsoft's GraphRAG approach (2024): (1) extract entities and relationships from chunks using an LLM, (2) build a graph and detect communities via hierarchical Leiden algorithm, (3) generate community summaries at multiple granularity levels, (4) at query time, retrieve via both vector search on chunks and graph traversal on communities. In Microsoft's evaluation, LLM evaluators preferred GraphRAG answers for comprehensiveness 72-83% of the time over standard vector RAG on global sensemaking queries. The trade-off is significantly higher indexing cost (many LLM calls for extraction + summarization), though LazyGraphRAG (2024) reduces this to 0.1% of the cost by deferring LLM use to query time.

**When GraphRAG helps**: Multi-hop reasoning, thematic/summarization queries across large corpora, datasets with rich entity relationships. **When to skip**: Simple factual lookups, small document sets, latency-constrained systems (graph construction is expensive).

### Multimodal retrieval

Text-only RAG fails for documents with charts, diagrams, tables, or images. Two approaches:

**CLIP-based retrieval**: Encode images and text into a shared embedding space. Query with text, retrieve relevant images (or vice versa). Works for image search but does not understand document layout.

**ColPali and vision-language models** (ICLR 2025): Encode entire document pages as images using a vision-language model (PaliGemma), skipping OCR and parsing entirely. ColPali uses a late-interaction architecture (like ColBERT) over ~1,030 patch vectors per page (1,024 image patches + instruction tokens), achieving strong retrieval on visually-rich documents like PDFs with tables, figures, and complex layouts. Successors include ColQwen2 (multi-resolution) and domain-specific variants. Trade-off: dramatically simpler indexing pipeline vs ~100x more vectors per page than single-vector models.

**When multimodal matters**: Document collections with significant visual content (scientific papers, financial reports, slide decks, manuals with diagrams). **When to skip**: Text-heavy corpora where standard chunking and embedding works well.

---

## Common pitfalls

**Using only dense retrieval**: Misses exact keyword matches. "HIPAA compliance" should strongly favor documents containing "HIPAA." Use hybrid search.

**Skipping reranking**: If your latency budget allows it, add reranking. Typically yields a meaningful MRR improvement.

**Reranking too few candidates**: Retrieve 50-100 in stage 1, rerank to 10-20, return top 5. Give the reranker room to correct stage 1 errors.

**Not tuning efSearch**: HNSW library defaults (typically 10-40, varying by library) prioritize speed over recall. Increase efSearch for better recall at the cost of latency; benchmark on your dataset.

**Returning low-scoring results**: If all top-k scores fall below a relevance threshold, return "no relevant information" instead of showing irrelevant chunks. Calibrate the threshold per embedding model and domain — score distributions vary significantly across models.

**Ignoring query length**: Single-word queries need expansion. 50+ word queries need decomposition.

**Not caching embeddings**: If a significant fraction of queries repeat, cache their embeddings to avoid redundant encoding.

**Filtering after retrieval**: Filter at query time using vector DB filters, not after retrieval.

**Using identical k for all queries**: Return variable k based on query complexity or confidence scores.

**Not monitoring retrieval independently**: Track Hit Rate@5 and MRR separately from generation quality to diagnose failures.

**Mixing incompatible score scales**: Always normalize to [0, 1] before combining dense and sparse scores.

**Over-optimizing stage 2**: If stage 1 recall is 70%, no reranking will fix it. Optimize stage 1 recall first.

**Applying MMR too early**: Rerank first (100 → 50), then apply MMR (50 → 5).

**Over-compressing context**: Start with threshold=0.5, monitor answer quality, adjust.

**Using sentence window for structured documents**: Use auto-merging retrieval instead for docs with cross-references or hierarchical structure.

**Parent-child retrieval without reranking**: Expansion increases tokens 5-10x. Always rerank after expansion.

**Inadequate RBAC metadata**: Add hierarchical RBAC fields at indexing time (roles, departments, owner_id), not just coarse access_level.

**Using ColBERT for <100 candidates**: Cross-encoder is fast enough at that scale, and ColBERT's precomputed-token speed advantage doesn't matter with so few candidates.

**Over-aggressive cache threshold (0.95-0.99)**: Barely better than exact-match. Use 0.90-0.95 for a meaningful improvement in hit rate.

**Caching without TTL**: Set domain-appropriate TTL. Add event-driven invalidation for critical accuracy.

---

## Production quality metrics & SLAs

Measurable success criteria for production RAG retrieval: quality benchmarks, latency SLAs, user satisfaction metrics, and cost trade-offs.

### Retrieval quality benchmarks

| Quality Level | Hit Rate@5 | Hit Rate@10 | When Acceptable |
|---------------|-----------|-------------|-----------------|
| **Minimum viable** | >70% | >80% | Prototype, internal beta |
| **Production ready** | >85% | >90% | Customer-facing, FAQ systems |
| **Best-in-class** | >95% | >98% | Legal, medical, compliance |

**MRR targets**: <0.5 needs improvement, 0.6-0.8 good, >0.85 excellent. Measure weekly on 200-500 labeled queries. See below for how to generate these.

### Synthetic test data & automated evaluation

Hand-labeling 500 queries is a major bottleneck. The standard approach: use LLMs to generate synthetic question-context pairs from your corpus, then evaluate retrieval automatically on every CI/CD run.

**Synthetic data generation**: Chunk your corpus → for each chunk, prompt an LLM to generate 1-3 questions that the chunk answers → you now have (question, ground_truth_chunk) pairs. Ragas and LlamaIndex both automate this, with Ragas adding "evolution" strategies (simple → reasoning → multi-context) to generate diverse difficulty levels.

```python
# Ragas synthetic test generation (API changes across versions — check docs)
from ragas.testset import TestsetGenerator

TESTSET_SIZE = 200
QUESTIONS_PER_CHUNK = 2

generator = TestsetGenerator(llm=wrapped_llm)
testset = generator.generate_with_langchain_docs(
    documents, testset_size=TESTSET_SIZE,
    # Query type distribution — class names vary by Ragas version:
    # v0.1: simple, reasoning, multi_context (from ragas.testset.evolutions)
    # v0.2+: SingleHopSpecificQuery, MultiHopAbstractQuery, MultiHopSpecificQuery
)

from llama_index.core.evaluation import generate_question_context_pairs

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=QUESTIONS_PER_CHUNK,
)
```

**Automated evaluation frameworks**:

| Framework | Key metrics | Unique strength | Cost |
|-----------|------------|-----------------|------|
| **Ragas** | Context precision, context recall, faithfulness, answer relevancy | End-to-end RAG eval + synthetic test generation | LLM calls (a few hundred queries per run) |
| **TruLens** | Context relevance, groundedness, answer relevance (RAG Triad) | Real-time dashboard, production monitoring | LLM calls (similar) |
| **ARES** (Stanford) | Prediction-powered inference | Statistically rigorous confidence intervals from fewer labels | Minimal LLM calls |
| **DeepEval** | 50+ metrics, G-Eval | CI/CD native (pytest plugin), unit test syntax | LLM calls |

**Retrieval-specific metrics**: **MRR@k** (reciprocal rank of first relevant result), **Hit Rate@k** (fraction of queries where at least one relevant result is in top k), **NDCG@k** (discounted cumulative gain normalized by ideal ranking). These measure retrieval quality independently of generation.

**CI/CD integration**: Run retrieval eval as a pipeline step — generate synthetic queries once (or refresh monthly), then compute MRR@10 and Hit Rate@5 on every deployment. A typical run evaluates 200 queries in 2-5 minutes. Set regression thresholds (e.g., fail if MRR drops >2% from baseline) to catch retrieval degradation before production.

### Latency SLAs

| Percentile | Target | Stage 1 | Stage 2 | Cache |
|-----------|--------|---------|---------|-------|
| p50 | <100ms | 30-50ms | 30-80ms | 5-15ms |
| p95 | <200ms | — | — | — |
| p99 | <500ms | — | — | — |
| Timeout | 1000ms | — | — | — |

### User satisfaction metrics

- **Answer helpfulness**: Target >4.0/5.0 average rating
- **Task completion**: Target >80% without escalating to support
- **Support deflection**: Target 30-50% ticket volume reduction
- **Query success rate**: Target >95% queries return non-empty results

### Monitoring dashboard

**Weekly**: Hit Rate@5, MRR, p50/p95/p99 latency, user rating, cache hit rate, error rate.
**Monthly report**: Quality trends, user satisfaction, system scale, cost, roadmap.
**Alert thresholds**: Hit Rate@5 <85%, p99 >500ms, error rate >5%, user rating <3.5/5.0 for >20% of queries.

### Quality vs cost trade-offs

| Use Case | Hit Rate@5 | Latency | Relative Cost |
|----------|-----------|---------|---------------|
| FAQ chatbot | >85% | Low | Lowest (simple pipeline, caching) |
| Enterprise search | >90% | Medium | Moderate (hybrid + reranking) |
| Legal/Medical | >95% | Medium | Highest (multi-stage, high accuracy) |
| Internal KB | >80% | Medium | Low (simpler pipeline, fewer queries) |

**Cost optimization**: semantic caching (significant cost reduction by avoiding redundant retrieval + generation), skip reranking for simple queries, hybrid search over dense-only (free quality gain), batch embeddings.

---

## Rollout & change management

Phased deployment strategy to minimize risk and measure impact: internal beta → limited production → full rollout → deprecate baseline.

### Phased rollout strategy

| Phase | Duration | Audience | Success Criteria |
|-------|----------|----------|-----------------|
| **1. Internal beta** | 2-4 weeks | 10-20 internal users | >80% correct, Hit Rate@5 >85%, p99 meets latency SLA |
| **2. Limited production** | 4-6 weeks | 10-25% users (A/B test) | Rating >3.8/5.0 AND >10% over baseline |
| **3. Full rollout** | 2-4 weeks | 25% → 50% → 75% → 100% | Stable at scale, p95 meets latency SLA, error <2% |
| **4. Deprecate baseline** | 30 days | Shadow mode only | Keep for emergency rollback, then decommission |

### A/B testing framework

- Split by user ID hash (consistent per-user experience)
- Minimum 1000 queries per group, recommended 5000+ for 2-3% effect detection
- Compare: user rating, task completion, p95 latency, cost per 1K queries

### Rollback plan

**Triggers**: Hit Rate@5 <75%, user satisfaction <3.0/5.0 for >30%, p95 latency exceeds SLA threshold sustained, error rate >10%.
**Mechanism**: Feature flag toggle (0% → 100% instant) or blue-green load balancer switch (<5 min).
**Post-rollback**: Root cause analysis within 24 hours, fix on staging, restart at phase 2.

### Change management

- **Internal**: Weekly status updates to PM, engineering, support
- **Support team**: Training on RAG (how it works, when to escalate, known limitations)
- **Users**: "Search is now semantic — understands intent, may differ from keyword search"
- **Engineering**: Expect 2-4 hours/week ongoing maintenance (index updates, monitoring, tuning)

---

## Operational risks & mitigation

RAG systems have failure modes that impact business outcomes. Key risks and mitigations for production operations.

### Infrastructure risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Vector DB failure** | System down, support tickets spike | 3-node cluster with failover, daily S3 backups, fallback to keyword search |
| **Embedding API outage** | Cannot embed queries | Exponential backoff retry, cached embeddings fallback, self-hosted backup model |
| **Index corruption** | Wrong results, user satisfaction drops | Daily snapshots, automated chunk count validation, regular DR drills (annual minimum, quarterly for critical systems) |

### Quality risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Quality degradation over time** | Hit Rate drifts as documents, users, or query patterns change | Weekly automated eval, document freshness monitoring, user feedback loop; if fine-tuning a custom embedding model, retrain on a regular cadence (quarterly is common) |
| **Embedding model change** | Must reindex all documents | Build new index in parallel, A/B test, hot-swap (no downtime) |

### Security risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Multi-tenant data leakage** | Severe contract, legal, and trust impact | RBAC hard filtering, quarterly pen testing, audit logging, regression tests |
| **PII/PHI exposure in logs** | Regulatory and compliance exposure | Sanitize logs (regex/NER), encrypt at rest + transit, access control, retention policy |

### Cost risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Unexpected cost spike** | Material budget overrun from query growth, reranking, or reindexing | Budget alerts, API rate limiting, per-user query quotas, incremental indexing |

### Business continuity

- **RTO**: <1 hour (vector DB restore from backup)
- **RPO**: <24 hours (daily backups)
- **Backup**: Vector DB snapshots + original docs (S3) + metadata (DB) + config (git), multi-region, 30-day retention
- **Incident response**: Detection (automated alerts) → Triage → Mitigation (rollback) → Recovery (restore) → Post-mortem
- **On-call**: 3-5 engineers rotating weekly; incident rate varies widely by system maturity and scale

---

## Best practices

**Default to two-stage retrieval.** Stage 1 retrieves 50-100 candidates quickly, stage 2 reranks to 5-10 with a meaningful MRR improvement. This covers the majority of use cases. For enterprise systems with strict accuracy requirements and generous latency budgets, consider three-stage retrieval (two-tower → ColBERT → cross-encoder) for additional accuracy gains.

**Default to hybrid search.** Dense + sparse with alpha=0.5. Adjust per query type.

**Optimize stage 1 for recall, stage 2 for precision.** Use efSearch=50-100 for HNSW. It's okay if stage 1 ranks the correct chunk 20th as long as it's in the top 50.

**Cache query embeddings for common queries.** In high-volume systems, a meaningful share of queries repeat — cache hits eliminate embedding inference latency entirely.

**Monitor retrieval independently of generation.** Track Hit Rate@5 and MRR separately. Diagnose whether failures are retrieval or generation problems.

**Use metadata filtering for explicit constraints only.** Don't filter by relevance score — let reranking handle that.

**Rerank at least 2x more candidates than you return.** LLM needs 5 chunks → retrieve 100, rerank to 50, return top 5.

**Profile latency at each stage.** Measure stage 1, stage 2, and total separately.

**Use query decomposition for multi-hop questions.** "Compare X and Y" needs separate retrievals.

**A/B test retrieval strategies.** If two-stage improves MRR by <5%, the complexity isn't justified.

**Apply MMR for multi-faceted queries, skip for precision search.** Redundancy confirms correctness in factual lookups.

**Apply compression after reranking, not before.** Only compress high-quality reranked candidates.

**Choose parent-child strategy by document structure.** Linear → sentence window (size 3). Hierarchical → auto-merging (threshold 0.5).

**Implement semantic caching for FAQ/support systems.** Significantly higher hit rate than exact-match caching, with substantial latency and cost savings.

**Use Matryoshka embeddings for large-scale indexes.** Truncate to 256-dim for stage 1 with minimal accuracy drop; let the cross-encoder recover precision in stage 2.

**Add self-querying when users specify constraints.** Dates, categories, and named entities in queries should become metadata filters, not embedding noise.

**Default to pipeline RAG; move to agentic when pipeline fails.** Pipeline RAG handles the majority of use cases with sub-second latency. Reserve agentic RAG for multi-hop reasoning and complex research queries.

**Automate retrieval evaluation with synthetic test data.** Generate 200-500 question-context pairs from your corpus, run MRR@10 and Hit Rate@5 on every deployment, set regression thresholds.
