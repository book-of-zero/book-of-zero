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
- [Common pitfalls](#common-pitfalls)
- [Production quality metrics & SLAs](#production-quality-metrics--slas)
- [Rollout & change management](#rollout--change-management)
- [Operational risks & mitigation](#operational-risks--mitigation)
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
- **Agentic retrieval**: Replacing fixed retrieval pipelines with an LLM agent that has retrieval as a tool. The agent decides when and how often to search, enabling iterative multi-hop retrieval at the cost of variable latency (1-30s vs predictable 200-500ms).
- **ANN (Approximate Nearest Neighbor)**: Algorithms (HNSW, IVF, PQ) that trade exact accuracy for sublinear search speed. Required past ~10K documents.
- **RBAC (Role-Based Access Control)**: Security filtering that physically blocks unauthorized chunks at query time via metadata filters. Distinct from soft boosting which only adjusts scores.
- **Semantic caching**: Caching retrieval results by query similarity (cosine >= 0.90-0.95) rather than exact string match. Achieves 40-70% hit rates in FAQ/support systems.

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

Three approaches to intent classification: rule-based (keyword matching, <1ms, for <10 intents), ML-based (embed queries + train classifier, for 10+ intents), and semantic similarity (few-shot matching against intent exemplars, no training needed).

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

intent_examples = {
    "greeting": ["Hi", "Hello", "Thanks", "Goodbye"],
    "knowledge_base": ["How do I reset my password?", "What is RAG?"],
    "structured_data": ["What's my account balance?", "Show my recent orders"],
    "real_time": ["What's the weather in SF?", "AAPL stock price"],
    "out_of_scope": ["Tell me a joke", "What's the meaning of life?"],
}

intent_embeddings = {
    intent: model.encode(examples)
    for intent, examples in intent_examples.items()
}

def classify_intent(query: str, threshold: float = 0.5) -> str:
    query_embedding = model.encode(query)
    intent_scores = {
        intent: float(util.cos_sim(query_embedding, embs)[0].max())
        for intent, embs in intent_embeddings.items()
    }
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] >= threshold else "knowledge_base"
```

**Rule-based (for small systems with <10 intents):**

```python
import re
from typing import Literal

IntentType = Literal["greeting", "knowledge_base", "structured_data", "real_time", "out_of_scope"]

def classify_intent_simple(query: str) -> IntentType:
    query_lower = query.lower()
    if re.search(r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b', query_lower):
        return "greeting"
    if re.search(r'\b(my account|my balance|my order|my subscription|purchase history)\b', query_lower):
        return "structured_data"
    if re.search(r'\b(weather|temperature|forecast|stock price|news)\b', query_lower):
        return "real_time"
    if re.search(r'\b(joke|poem|story|write code|meaning of life)\b', query_lower):
        return "out_of_scope"
    return "knowledge_base"
```

**ML-based (for 10+ intents or nuanced queries):**

```python
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

training_data = [
    ("Hi", "greeting"), ("How do I reset my password?", "knowledge_base"),
    ("What's my account balance?", "structured_data"),
    ("What's the weather in SF?", "real_time"),
    ("Tell me a joke", "out_of_scope"),
    # ... 100-1000+ examples
]

queries, intents = zip(*training_data)
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode(queries)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(query_embeddings, intents)

def classify_intent_ml(query: str) -> str:
    return classifier.predict([model.encode([query])[0]])[0]
```

**Routing logic:**

```python
def route_query(query: str, user_id: str) -> dict:
    intent = classify_intent(query)
    if intent == "greeting":
        return {"intent": intent, "response": "Hi! How can I help?", "source": "direct"}
    elif intent == "knowledge_base":
        chunks = hybrid_search(query, documents, k=5)
        return {"intent": intent, "response": llm_generate(query, chunks), "source": "vector_db"}
    elif intent == "structured_data":
        result = database.execute(generate_sql(query, user_id))
        return {"intent": intent, "response": format_sql_result(result), "source": "sql"}
    elif intent == "real_time":
        return {"intent": intent, "response": call_api(query), "source": "api"}
    else:
        return {"intent": "out_of_scope", "response": "I'm designed for account and product support.", "source": "direct"}
```

**Multi-step routing** for complex queries needing multiple backends (e.g., "Show my recent orders and recommend similar products"): use an LLM to decompose the query into steps, classify and route each step independently, then combine results.

**When routing matters**: Multi-backend systems (RAG + SQL + APIs), customer support chatbots, enterprise assistants. Less important for single-backend systems or specialized tools. Cost: 5-20ms latency (semantic) or <1ms (rule-based) to avoid 30-60% of useless retrievals.

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

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Index time: encode all documents once
doc_embeddings = model.encode(documents, convert_to_numpy=True)
index = faiss.IndexHNSWFlat(doc_embeddings.shape[1], 32)
index.add(doc_embeddings)

# Search time: encode query, find nearest neighbors
query_embedding = model.encode("What is RAG?", convert_to_numpy=True)
distances, indices = index.search(query_embedding[np.newaxis, :], k=50)
```

The landmark two-tower architecture from Facebook AI Research. Uses separate BERT-based encoders for queries and contexts, trained with contrastive loss.

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
c_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

doc_tokens = c_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    doc_embeddings = c_encoder(**doc_tokens).pooler_output.numpy()

query_tokens = q_tokenizer([query], return_tensors="pt")
with torch.no_grad():
    query_embedding = q_encoder(**query_tokens).pooler_output.numpy()

similarities = np.dot(doc_embeddings, query_embedding.T).squeeze()
top_indices = np.argsort(similarities)[::-1][:50]
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

client = OpenAI()

# Full 3072-dim embedding
full = client.embeddings.create(input="What is RAG?", model="text-embedding-3-large")

# Truncated 256-dim — API handles it natively
compact = client.embeddings.create(
    input="What is RAG?", model="text-embedding-3-large", dimensions=256
)
# For models without native truncation: embedding[:256] then L2-normalize
```

**MRL in two-stage retrieval**: Use 256-dim embeddings for stage 1 (fast ANN recall over millions of documents), then let the cross-encoder handle precision in stage 2. The accuracy drop from truncation is typically smaller than the cross-encoder's improvement — so the pipeline as a whole loses almost nothing while stage 1 becomes significantly faster and cheaper.

**Vector DB support**: Milvus (v2.5+ "Funnel Search" — shortlist with truncated dims, rescore with full dims in one query), Qdrant (prefetch + rescore via Query API), Weaviate (configurable `dimensions` for OpenAI, named vectors for multi-resolution). Pinecone, pgvector, Chroma accept any dimension but require application-level two-stage logic.

**When MRL helps**: Large-scale indexes (>1M documents) where storage and ANN latency matter, or when you want to reduce embedding API costs. **When to skip**: Small indexes where full-dimension search is already fast enough, or when you need maximum stage 1 precision without reranking.

### BM25 sparse retrieval

BM25 scores documents by term frequency (TF) and inverse document frequency (IDF), with saturation to prevent over-weighting of repeated terms.

```python
from rank_bm25 import BM25Okapi
import numpy as np

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

scores = bm25.get_scores("HIPAA compliance requirements".lower().split())
top_indices = np.argsort(scores)[::-1][:50]
```

**How BM25 works**: TF rewards more term occurrences with saturation (10th occurrence adds less than 1st). IDF weights rare terms higher ("HIPAA" beats "the"). Document length normalization prevents unfair penalization of long documents. Key parameters: k1 (term saturation, default 1.5) and b (length normalization, default 0.75).

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
| **SPLADE++ (EnsembleDistil)** | 38.0% | High | 30-45ms |
| **SPLADE-v3** | 40.2% | High | 30-45ms |

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

def encode_splade(text: str) -> dict:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**tokens).logits
    # Max pooling over sequence length, then ReLU + log(1+x) for sparsity
    weights = torch.max(torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1).values
    # Extract non-zero terms
    non_zero = weights.squeeze().nonzero().squeeze()
    return {tokenizer.decode(idx.item()): weights[0, idx].item() for idx in non_zero}
```

**When SPLADE works**: Hybrid setups where you want semantic sparse retrieval without a separate dense index. Particularly strong for domain-specific vocabulary and abbreviations. **When to skip**: If you already run hybrid dense + BM25 and don't want to retrain; or if sub-5ms latency is required (BM25 is faster).

**Infrastructure**: SPLADE outputs are sparse vectors compatible with inverted indexes. Qdrant (native sparse vectors + built-in SPLADE via FastEmbed), Vespa (native SPLADE embedder), Milvus (sparse vectors in same collection as dense), and Pinecone (sparse-dense hybrid) all support sparse vector search. Weaviate uses BM25 for its sparse component and does not natively support SPLADE vectors.

### Hybrid search

Combine dense and sparse retrieval to get the best of both: semantic understanding and keyword precision.

```python
def hybrid_search(
    query: str,
    documents: list[str],
    doc_embeddings: np.ndarray,
    k: int = 50,
    alpha: float = 0.5,
):
    # Dense retrieval
    query_embedding = model.encode(query, convert_to_numpy=True)
    dense_scores = np.dot(doc_embeddings, query_embedding)

    # Sparse retrieval
    sparse_scores = bm25.get_scores(query.lower().split())

    # Normalize both to [0, 1] — critical for fair combination
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
    sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

    hybrid_scores = alpha * dense_norm + (1 - alpha) * sparse_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    return [(i, hybrid_scores[i]) for i in top_indices]
```

Precompute `doc_embeddings` once at index time. Search-time work should encode only the query, score against stored vectors, and fuse with sparse scores.

**Tuning alpha**: alpha=1.0 pure dense, alpha=0.0 pure sparse, alpha=0.5 balanced default, alpha=0.7 favor semantic (conversational), alpha=0.3 favor keywords (technical/entity search). Hybrid typically improves Hit Rate@10 by 5-15% over single-method.

**Alternative: Reciprocal Rank Fusion (RRF)** merges ranked lists without score normalization: `score(doc) = sum(1 / (k + rank))` with k=60. More robust than score combination but less tunable. Use RRF when you don't have time to tune alpha.

```python
def reciprocal_rank_fusion(dense_ranks: list[int], sparse_ranks: list[int], k: int = 60):
    scores = {}
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

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Stage 1: Get 50 candidates
candidates = hybrid_search(query, documents, k=50)
candidate_docs = [documents[i] for i, score in candidates]

# Stage 2: Rerank
pairs = [[query, doc] for doc in candidate_docs]
rerank_scores = reranker.predict(pairs)
top_5_docs = [candidate_docs[i] for i in np.argsort(rerank_scores)[::-1][:5]]
```

**Models**: ms-marco-MiniLM-L-6-v2 (fast, 6 layers, default), ms-marco-MiniLM-L-12-v2 (better quality, 2x slower), mmarco-mMiniLMv2-L12-H384-v1 (multilingual, 100+ languages).

**When reranking matters**: Stage 1 has high recall but low precision (correct chunk in top 50 but ranked 20th), or queries require nuanced understanding ("compare X and Y"). Typically improves MRR by 10-20%. **Skip reranking** when latency budget is <100ms total or stage 1 already achieves >95% Hit Rate@5.

**Latency**: Cross-encoder latency depends heavily on model size and candidate count. With ms-marco-MiniLM-L-6-v2 reranking 50 candidates: 30-50ms. With larger models or 500 candidates: 50-500ms. Budget accordingly — a two-stage pipeline with a small cross-encoder reranking 50 candidates typically runs at 60-100ms total (stage 1 + stage 2).

### ColBERT (Late Interaction)

ColBERT bridges the accuracy-latency gap between two-tower and cross-encoder. It encodes queries and documents independently at **token-level granularity**, then computes similarity via **MaxSim** (for each query token, find max similarity with any doc token, sum the scores).

```
Two-tower:  Query → [single 768-dim vector], Doc → [single 768-dim vector]
ColBERT:    Query → [N × 128-dim vectors],  Doc → [M × 128-dim vectors]
            Similarity = Σ max(cos(Qi, Dj)) for all query tokens Qi
```

| Architecture | Candidates | Latency | MRR@10 | Storage |
|--------------|------------|---------|---------|---------|
| **Bi-encoder** | 1M | 1-5ms | 35-37% | 1x (768-dim/doc) |
| **ColBERT** | 100-1000 | 10-50ms | 40.8% | 6-10x (128-dim/token) |
| **Cross-encoder** | 10-50 | 50-500ms (model & k dependent) | 42-44% | Model params only |

```python
from ragatouille import RAGPretrainedModel

colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Three-stage: two-tower (1M → 100) → ColBERT (100 → 50) → cross-encoder (50 → 5)
stage1_results = dense_retrieval(query, k=100)
stage2_results = colbert_model.rerank(query=query, documents=stage1_results, k=50)
stage3_results = cross_encoder.rerank(query, stage2_results, k=5)
```

**Use ColBERT** when reranking 100-1000 candidates where cross-encoder is too slow. **Skip ColBERT** for <100 candidates (cross-encoder is fast enough) or when 6-10x storage overhead is unacceptable.

**Compression**: PLAID (centroid IDs + residuals, 2-3x smaller) and WARP (weighted aggregation, 2x smaller) reduce storage with <1% accuracy loss.

### LLM-based rerankers

Cross-encoders use BERT-scale models (110M-340M parameters). LLM-based rerankers use larger language models for higher accuracy at the cost of latency and compute.

| Reranker | Approach | NDCG@10 (BEIR avg) | Latency (50 docs) | Cost |
|----------|----------|-------------------|-------------------|------|
| **ms-marco-MiniLM-L-6-v2** | Cross-encoder (22M) | ~49% | 30-50ms | Self-hosted |
| **BGE Reranker v2-m3** | Cross-encoder (0.6B) | ~66% | 50-100ms | Self-hosted |
| **Cohere Rerank 3.5** | Purpose-built (API) | ~56% | ~400ms | ~$2/1K searches |
| **RankLLaMA-13B** | Fine-tuned LLaMA | ~59% (DL19) | 200-500ms | Self-hosted (GPU) |
| **RankGPT (GPT-4o)** | Listwise via LLM | ~75% (DL19) | 1-3s | API cost (high) |

**How they differ**: Cross-encoders (MiniLM, BGE) score each query-document pair independently (pointwise) and are purpose-built for relevance scoring — up to 60x cheaper and 48x faster than general-purpose LLM rerankers. RankLLaMA fine-tunes a 13B LLM for pointwise scoring. RankGPT passes all candidates to GPT-4 and asks it to rank them as a list (listwise), achieving the highest accuracy but at significant cost.

**When to use LLM-based rerankers**: High-stakes domains (legal, medical, compliance) where accuracy improvement justifies the latency and cost. For most production systems, purpose-built rerankers (BGE, Cohere) offer the best price-performance ratio. **When to use RankGPT/RankLLaMA**: Research, low-volume high-stakes applications, or when cross-encoder accuracy is insufficient. **When to skip entirely**: Latency budget <100ms or cost-sensitive workloads.

---

## Search-time index tuning

Vector indexes are built during indexing (see [RAG Indexing Vector Storage]({{ site.baseurl }}/docs/genai/rag/indexing/page/#vector-storage)). At search time, tune parameters to balance recall and latency.

```python
index = faiss.read_index("index.faiss")
index.hnsw.efSearch = 50  # Default 16, increase for better recall

# efSearch=16: 90% recall, <5ms | efSearch=50: 95% recall, <10ms | efSearch=100: 98% recall, <20ms
```

Set efSearch=100 when high stage 1 recall is critical (missing the correct chunk means it won't be reranked). Set efSearch=20 when latency is tight and you're reranking 100+ candidates to compensate.

---

## Search-time filtering & boosting

Filtering and boosting happen during search execution — they are not query rewrites but constraints and score adjustments applied to the retrieval process itself.

### Metadata filtering

Filter by document metadata before retrieval to reduce search space and improve precision.

```python
results = client.search(
    collection_name="documents",
    query_vector=query_embedding.tolist(),
    query_filter=Filter(must=[
        FieldCondition(key="category", match=MatchValue(value="healthcare")),
        FieldCondition(key="created_at", range={"gte": "2024-01-01"}),
    ]),
    limit=50,
)
```

Helps when users specify constraints or for time-sensitive queries. Hurts when filters are too restrictive or metadata is incomplete.

### Hard security filtering (RBAC)

Hard security filtering physically blocks unauthorized chunks from retrieval results. This is distinct from score boosting — unauthorized users never see restricted content, period. Critical for enterprise RAG handling sensitive data.

```python
def secure_retrieval(query: str, user_token: str, k: int = 10):
    payload = jwt.decode(user_token, SECRET_KEY, algorithms=["HS256"])
    user_roles = payload.get("roles", [])

    rbac_filter = Filter(must=[
        FieldCondition(key="roles", match=MatchAny(any=user_roles))
    ])

    results = client.search(
        collection_name="documents",
        query_vector=model.encode(query).tolist(),
        query_filter=rbac_filter,
        limit=k,
    )
    return [{"text": hit.payload["text"], "score": hit.score} for hit in results]
```

**You must implement RBAC** for multi-tenant SaaS, regulated industries (HIPAA, GDPR, SOX, PCI-DSS), sensitive data types (PII, PHI, financial, legal), and enterprise contracts requiring SOC 2 or ISO 27001. **Soft boosting is sufficient** for single-tenant systems with public data only and <10 trusted users.

**Business value**:

| Item | Amount |
|------|--------|
| Implementation cost | $50K-$150K (identity integration, metadata audit, monitoring) |
| Annual operational cost | $10K-$30K |
| Value protected: breach prevention | $2M-$5M+ (average breach cost avoided) |
| Value protected: compliance | $100K-$20M (HIPAA/GDPR penalties avoided) |
| Payback period | <6 months for regulated industries |

**Concrete scenarios**:
- **Healthcare (HIPAA)**: Cardiologist blocked from psychiatry records. Violation = $100-$1.5M penalty + criminal charges.
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
def secure_retrieval_with_audit(query: str, user_token: str, k: int = 10):
    payload = jwt.decode(user_token, SECRET_KEY, algorithms=["HS256"])
    user_id, user_roles = payload["user_id"], payload.get("roles", [])

    results = client.search(
        collection_name="documents",
        query_vector=model.encode(query).tolist(),
        query_filter=Filter(must=[FieldCondition(key="roles", match=MatchAny(any=user_roles))]),
        limit=k,
    )

    audit_log = {
        "timestamp": datetime.utcnow().isoformat(),
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
- **SOC 2 Type II**: 12-month access control enforcement metrics
- **HIPAA § 164.308(a)(4)**: Audit logs with user ID, timestamp, PHI accessed, authorization basis
- **GDPR Article 30**: Records of data processing activities including access controls
- **ISO 27001 A.9.1**: Policy document + technical implementation + enforcement audit logs

**Technical trade-offs**: +5-15ms latency, +5-10% storage for RBAC metadata. Always validate JWT signatures and expiration. Sync permissions regularly between IdP and vector DB or use short-lived tokens (1-hour TTL).

**Pitfalls**: Missing RBAC metadata at indexing time forces reindexing. Trusting unvalidated JWT tokens is a critical vulnerability. Permission drift between IdP and vector DB causes stale access — sync regularly.

**Cross-references**: See [indexing metadata strategy]({{ site.baseurl }}/docs/genai/rag/indexing/page/#metadata-strategy) for structuring RBAC fields at indexing time.

---

### Score boosting

Adjust retrieval scores based on metadata to promote authoritative, recent, or contextually relevant documents.

```python
def metadata_boosted_search(query: str, documents: list[dict], k: int = 10):
    base_scores = np.dot(doc_embeddings, model.encode(query))
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + 1e-8)

    for i, doc in enumerate(documents):
        # Recency: exponential decay over 1 year, up to +20%
        age_days = (datetime.utcnow() - datetime.fromisoformat(doc['created_at'])).days
        base_scores[i] *= (1 + 0.2 * np.exp(-age_days / 365))
        # Authority: trusted sources +10%
        if doc.get('author') in ['legal-team', 'compliance-team']:
            base_scores[i] *= 1.1
        # Context: match user's department +15%
        if doc.get('category') == user_context['department']:
            base_scores[i] *= 1.15

    return [documents[i] for i in np.argsort(base_scores)[::-1][:k]]
```

Keep boosts modest (10-30%). A recent irrelevant doc shouldn't beat an older highly relevant doc.

### Positional boosting

Introduction and conclusion sections often contain key concepts and summaries. Boost chunks based on position to improve precision for conceptual queries.

**Lost in the middle**: LLMs exhibit U-shaped performance — >20% drop when relevant information is in the middle of long contexts. This affects both which chunks to retrieve and how to order them for the LLM.

```python
# U-shaped boost: first chunk +20%, last chunk +15%, section headers +10%
# Middle chunks of long docs get slight penalty

# After retrieval, reorder chunks for LLM input:
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place most relevant chunks at start and end, least relevant in middle."""
    reordered = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered.insert(i // 2, chunk)
        else:
            reordered.append(chunk)
    return reordered
```

```python
def positional_boosted_search(query: str, documents: list[dict], k: int = 10):
    base_scores = np.dot(doc_embeddings, model.encode(query))
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + 1e-8)
    boosted_scores = base_scores.copy()

    for i, doc in enumerate(documents):
        chunk_position = doc.get('chunk_index', 0)
        total_chunks = doc.get('total_chunks', 1)

        if chunk_position == 0:
            boosted_scores[i] *= 1.2
        elif chunk_position == total_chunks - 1:
            boosted_scores[i] *= 1.15
        if doc.get('is_section_header', False):
            boosted_scores[i] *= 1.1
        if total_chunks > 10:
            normalized_position = chunk_position / (total_chunks - 1)
            position_weight = 1 - 0.15 * (1 - 4 * (normalized_position - 0.5) ** 2)
            boosted_scores[i] *= position_weight

    return [documents[i] for i in np.argsort(boosted_scores)[::-1][:k]]

SECTION_WEIGHTS = {
    'research_paper': {'abstract': 1.15, 'methodology': 1.25, 'results': 1.2, 'references': 0.8},
    'technical_doc': {'implementation': 1.3, 'api_reference': 1.2, 'troubleshooting': 1.15},
    'legal_doc': {'definitions': 1.3, 'requirements': 1.25, 'obligations': 1.25, 'preamble': 0.9},
}
```

Helps for structured documents (papers, reports, technical docs). Hurts for unstructured content (chat logs, transcripts) or queries seeking specific details in middle sections. Store positional metadata during indexing. Combine with reranking: apply positional boosts before cross-encoder, not after.

---

## Query optimization

Queries are not always well-formed. Users ask "it", "more on that", or complex multi-part questions. Query optimization rewrites queries before retrieval.

### Query expansion

Add synonyms or related terms to increase recall. Use T5 or similar to generate paraphrases, concatenate with original query for retrieval.

```python
def expand_query(query: str) -> str:
    input_ids = tokenizer(f"paraphrase: {query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=3, num_beams=5)
    paraphrases = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return query + " " + " ".join(paraphrases)
```

Helps for short/ambiguous queries and domain abbreviations ("ML" → "machine learning"). Hurts for long, specific queries where expansions add noise.

### Conversational query rewriting

In multi-turn conversations, follow-up queries like "What are its limitations?" lack context. Rewriting reformulates them into standalone questions using chat history.

```python
def rewrite_conversational_query(query: str, chat_history: list[dict]) -> str:
    recent_history = chat_history[-5:]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
    prompt = f"""Rewrite the user's latest query into a standalone question.

Conversation history:
{history_text}

Latest query: {query}

Standalone query:"""
    return llm.invoke(prompt).content.strip()
```

**Detecting when rewriting is needed**: Check for pronouns and contextual references (it, its, this, that, also, instead). Skip rewriting for standalone queries to save 100-300ms.

**Latency optimization**: Use a smaller model (GPT-3.5, Claude Haiku: 50-100ms vs 300ms), cache rewritten queries by (query + history hash), or fine-tune T5-small for 10-20ms rewrites at 90-95% of GPT-4 quality.

```python
import hashlib, json, re

def needs_rewriting(query: str) -> bool:
    patterns = [
        r'\b(it|its|this|that|these|those|they|them|their)\b',
        r'\b(also|too|as well|additionally|furthermore)\b',
        r'\b(instead|however|but|although)\b',
        r'\b(the same|similar|different|compared)\b',
    ]
    return any(re.search(p, query, re.IGNORECASE) for p in patterns)

def get_chat_history_hash(chat_history: list[dict]) -> str:
    return hashlib.md5(json.dumps(chat_history, sort_keys=True).encode()).hexdigest()

def rewrite_with_cache(query: str, chat_history: list[dict], cache: dict) -> str:
    cache_key = f"{query}:{get_chat_history_hash(chat_history)}"
    if cache_key in cache:
        return cache[cache_key]
    rewritten = rewrite_conversational_query(query, chat_history)
    cache[cache_key] = rewritten
    return rewritten
```

**Fine-tuned rewriting model** (10-20ms vs 100-300ms for GPT-4):

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("your-org/t5-query-rewriter")
tokenizer = T5Tokenizer.from_pretrained("your-org/t5-query-rewriter")

def rewrite_fast(query: str, chat_history: list[dict]) -> str:
    history_text = " ".join([msg['content'] for msg in chat_history[-3:]])
    input_ids = tokenizer(f"rewrite query with context: {history_text} [SEP] {query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Critical for** chatbot interfaces, multi-turn Q&A, and voice assistants. **Less important for** single-turn search, keyword search, and FAQ systems.

### Query decomposition

Break multi-part questions into sub-queries, retrieve separately, merge and rerank with the original query.

```python
def decompose_query(query: str) -> list[str]:
    prompt = f"Break this into 2-4 simple sub-questions:\n\nQuestion: {query}\n\nSub-questions:"
    return [q.strip() for q in llm.invoke(prompt).content.split("\n") if q.strip()]

# "Compare HIPAA and GDPR for healthcare data" →
# 1. What are HIPAA's privacy requirements?
# 2. What are GDPR's privacy requirements?
# 3. How do they differ for healthcare data?

all_results = []
for sub_q in decompose_query(query):
    all_results.extend(hybrid_search(sub_q, documents, k=20))
top_chunks = rerank(query, list(set(all_results)), k=5)
```

Helps for multi-hop questions ("compare X and Y"). Hurts for simple queries where decomposition adds 200-500ms overhead.

### Self-querying (natural language to metadata filters)

When a user asks "Show me healthcare policies from 2023", running semantic search on the full text is inefficient — "2023" will pollute the embedding. **Self-querying** uses an LLM (or a smaller model) to extract structured metadata filters from the query text, then passes only the semantic portion to the vector search.

```
User query: "healthcare policies from 2023"
    ↓ LLM extraction
Semantic query: "policies"
Metadata filters: {category: "healthcare", year: >= 2023}
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
#   query="policies", filter=AND(eq("category","healthcare"), gte("year",2023))
```

The LLM receives a structured prompt describing available metadata fields (names, types, descriptions) and outputs a structured query with a semantic text component and filter expressions. LangChain's `SelfQueryRetriever` and LlamaIndex's `VectorIndexAutoRetriever` both implement this pattern.

**Lightweight alternatives**: For predictable metadata patterns, skip the LLM call entirely. Use regex or NER models to extract dates, categories, and named entities — 1-5ms vs 100-500ms for an LLM call. Reserve LLM-based extraction for open-ended metadata schemas or complex filter logic (nested AND/OR conditions).

**When self-querying helps**: Queries with explicit constraints (dates, categories, authors, document types) that map to indexed metadata fields. **When to skip**: Purely semantic queries without metadata constraints, or when metadata fields are sparse/unreliable. **Latency**: 100-500ms for LLM-based, 1-5ms for regex/NER-based extraction.

### Advanced query techniques

#### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, embed it instead of the raw query. Documents are written as answers — matching answer-to-answer is more effective than matching question-to-answer.

```python
def hyde_retrieval(query: str, documents: list[str], k: int = 5):
    hypothetical_answer = llm.invoke(
        f'Given the question: "{query}"\nWrite a detailed, factual answer.'
    ).content
    hyde_embedding = model.encode(hypothetical_answer)
    similarities = np.dot(model.encode(documents), hyde_embedding)
    return [documents[i] for i in np.argsort(similarities)[::-1][:k]]
```

Helps for complex queries with large semantic gaps between query and document vocabulary. Hurts when the hypothetical answer is wrong (leads retrieval astray) or for simple queries. Adds 1-3s LLM latency. Optimization: average 3-5 hypothetical answer embeddings for robustness.

#### Multi-Query

Generate multiple query variations from different perspectives, retrieve for all, merge with RRF.

```python
def multi_query_retrieval(query: str, documents: list[str], k: int = 5):
    variations = llm.invoke(
        f"Generate 3 different versions of: {query}"
    ).content.split("\n")[:3]

    scores = {}
    for q in [query] + variations:
        ranked = np.argsort(np.dot(model.encode(documents), model.encode(q)))[::-1][:50]
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    return [documents[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]
```

Helps for ambiguous queries and domain-specific terminology variation. Adds 1-2s LLM latency + multiple retrievals.

#### Step-back prompting

Generate a more general query to retrieve foundational context alongside specific details.

```python
def step_back_retrieval(query: str, documents: list[str], k: int = 10):
    step_back_query = llm.invoke(
        f'Generate a more general question for: "{query}"'
    ).content.strip()

    specific = hybrid_search(query, documents, k=k)
    general = hybrid_search(step_back_query, documents, k=k)
    combined = specific[:int(k * 0.7)] + general[:int(k * 0.3)]
    return rerank(query, list({doc['chunk_id']: doc for doc in combined}.values()), k=k)
```

Helps for complex domain questions requiring background context. Hurts for simple factual queries. Adds 500ms-1s LLM latency.

---

## Result optimization

Query optimization transforms the query before retrieval. Result optimization processes retrieved chunks to improve diversity, reduce redundancy, or expand context.

### MMR (Maximal Marginal Relevance)

Standard retrieval can return 5 chunks discussing the same narrow aspect. MMR balances relevance and diversity: `MMR = lambda * sim(query, chunk) - (1-lambda) * max(sim(chunk, selected_chunks))`.

```python
def mmr_rerank(query: str, candidates: list[str], k: int = 5, lambda_param: float = 0.5):
    query_emb = model.encode(query)
    cand_embs = model.encode(candidates)
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

**Tuning lambda**: 0.7-0.9 favor relevance (precision search), 0.5 balanced default, 0.3-0.5 favor diversity (exploratory/multi-faceted queries). Apply MMR after reranking (100 → 50 with cross-encoder, then MMR 50 → 5). Computational cost scales O(k^2).

### Contextual compression

Not every sentence in a chunk is relevant. Compression filters chunks to keep only query-relevant content.

**Embedding-based filtering** (fast, 10-50ms): re-embed sentences, keep those above similarity threshold.

```python
def embedding_compress(query: str, chunks: list[str], threshold: float = 0.5) -> list[str]:
    query_emb = model.encode(query)
    compressed = []
    for chunk in chunks:
        sentences = chunk.split('. ')
        relevant = [s for s, sim in zip(sentences, np.dot(model.encode(sentences), query_emb)) if sim >= threshold]
        if relevant:
            compressed.append('. '.join(relevant))
    return compressed
```

**LLM-based extraction** (high quality, 100-500ms/chunk): prompt LLM to extract relevant sentences verbatim. 4-15x compression but expensive at scale.

```python
def llm_compress(query: str, chunks: list[str]) -> list[str]:
    prompt_template = """Extract ONLY sentences relevant to the query. Return empty if none.
Query: {query}
Chunk: {chunk}
Relevant sentences:"""
    return [r for chunk in chunks
            if (r := llm.invoke(prompt_template.format(query=query, chunk=chunk)).content.strip())]

def pipeline_compress(query: str, chunks: list[str], redundancy_threshold: float = 0.85, relevance_threshold: float = 0.5):
    query_emb = model.encode(query)
    all_sentences = [s.strip() for chunk in chunks for s in chunk.split('. ') if s.strip()]
    sent_embs = model.encode(all_sentences)

    # Remove redundant sentences
    unique_sents, unique_embs = [], []
    for sent, emb in zip(all_sentences, sent_embs):
        if not unique_embs or np.max(np.dot(unique_embs, emb)) < redundancy_threshold:
            unique_sents.append(sent)
            unique_embs.append(emb)

    # Filter by relevance
    scores = np.dot(unique_embs, query_emb)
    return [s for s, score in zip(unique_sents, scores) if score >= relevance_threshold]
```

**Threshold tuning**: 0.3 permissive (2x compression), 0.5 balanced (3-4x), 0.7 aggressive (5-10x, risks removing important context). Apply after reranking, not before. Start with embedding-based; use LLM-based only when precision is critical.

### Parent-child retrieval

Small chunks match queries precisely but lack context. Large chunks provide context but match poorly. Parent-child retrieval solves this: retrieve based on small chunks, return large chunks to the LLM.

**Sentence window retrieval**: Embed individual sentences. When retrieved, expand to +/-N surrounding sentences (default N=5, yielding 11 sentences). Simple, works for linear documents.

**Auto-merging retrieval**: Build a hierarchical tree (parent 2048 tokens → intermediate 512 → leaf 128). Retrieve leaf nodes; if 60%+ of a parent's children are retrieved, return the parent instead. Better for structured documents.

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
    def __init__(self, documents: list[str], window_size: int = 5):
        self.window_size = window_size
        self.sentences, self.metadata = [], []
        for doc_idx, doc in enumerate(documents):
            doc_sents = [s.strip() + '.' for s in doc.split('.') if s.strip()]
            for sent_idx, sent in enumerate(doc_sents):
                self.sentences.append(sent)
                self.metadata.append({'doc_idx': doc_idx, 'sent_idx': sent_idx, 'doc_sents': doc_sents})
        self.embeddings = model.encode(self.sentences)

    def search(self, query: str, k: int = 5):
        sims = np.dot(self.embeddings, model.encode(query))
        windows = []
        for idx in np.argsort(sims)[::-1][:k]:
            m = self.metadata[idx]
            start = max(0, m['sent_idx'] - self.window_size)
            end = min(len(m['doc_sents']), m['sent_idx'] + self.window_size + 1)
            windows.append({'window': ' '.join(m['doc_sents'][start:end]), 'target': self.sentences[idx]})
        return windows
```

**Auto-merging:**

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HierarchicalNode:
    chunk_id: str
    text: str
    level: int  # 0=parent, 1=intermediate, 2=leaf
    parent_id: Optional[str]
    children_ids: List[str]

def auto_merge_retrieve(query: str, nodes: dict, k: int = 12, merge_threshold: float = 0.6):
    leaf_nodes = [n for n in nodes.values() if n.level == 2]
    leaf_embs = model.encode([n.text for n in leaf_nodes])
    top_leaves = [leaf_nodes[i] for i in np.argsort(np.dot(leaf_embs, model.encode(query)))[::-1][:k]]

    parent_to_children = {}
    for leaf in top_leaves:
        parent_to_children.setdefault(leaf.parent_id, []).append(leaf.chunk_id)

    final = []
    for parent_id, child_ids in parent_to_children.items():
        parent = nodes[parent_id]
        if len(child_ids) / len(parent.children_ids) >= merge_threshold:
            final.append(parent.text)
        else:
            final.extend(nodes[cid].text for cid in child_ids)
    return final
```

**Tuning**: Window size 3 gives significant improvement with reasonable tokens; 5 is a safe default; >5 has diminishing returns. Merge threshold 0.5 aggressive, 0.6 balanced, 0.7-0.8 conservative. Always apply reranking after expansion to filter the best expanded contexts.

### Document-level vs chunk-level retrieval

Sometimes return entire documents instead of chunks. Short documents (<4K tokens), single-purpose content, and queries requiring narrative coherence benefit from document-level retrieval.

**Hybrid approach**: Index small chunks (400 tokens) for precise matching, retrieve and return larger parent chunks (2000 tokens) or full documents to the LLM.

```python
# LangChain parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=50),
    parent_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200),
)
# Queries search child chunks but return parent chunks
```

| Aspect | Chunk-Level | Document-Level | Parent Document |
|--------|-------------|----------------|-----------------|
| **Precision** | High | Low | High (child chunks) |
| **Context** | Low | High | High (parent chunks) |
| **Token Cost** | Low | High | Medium-High |

**Sizing rule**: <4K tokens → no chunking, 4K-10K → parent document retrieval, >10K → standard chunking. Route summarization queries to document-level, specific questions to chunk-level.

### Context enrichment

Pass metadata to the LLM for attribution. Include source, page, date, author — enough for users to verify. Prompt the LLM to cite sources explicitly ([1], [2]).

```python
def retrieve_with_metadata(query: str, k: int = 5):
    results = client.search(collection_name="documents", query_vector=model.encode(query).tolist(), limit=k)
    context = []
    for i, r in enumerate(results):
        m = r.payload
        context.append(f"[Source {i+1}] {m.get('source','?')} (p.{m.get('page_number','?')}, {m.get('created_at','?')})\n{r.payload['text']}")
    return f"Answer using only these sources. Cite by number.\n\n{''.join(context)}\n\nQuestion: {query}"
```

---

## Semantic caching

Caching retrieval results by query similarity rather than exact string match is a major architectural optimization. **Semantic caching** compares new queries against cached query embeddings (cosine >= 0.90-0.95), handling paraphrased queries and achieving 40-70% hit rates vs 10-20% for exact-match caching.

Semantic caching is typically the first routing intercept — check the cache before any retrieval work.

```python
class SemanticCache:
    def __init__(self, redis_client, encoder, threshold=0.92, ttl=3600):
        self.redis, self.encoder = redis_client, encoder
        self.threshold, self.ttl = threshold, ttl

    def get(self, query: str):
        query_emb = self.encoder.encode(query)
        # Compare against all cached query embeddings
        # Return cached results if similarity >= threshold
        # Return None on miss

    def set(self, query: str, results: list):
        # Store query embedding + results with TTL

def retrieve_with_cache(query: str, **kwargs):
    cached = cache.get(query)
    if cached:
        return {"results": cached, "from_cache": True, "latency_ms": 12}
    response = retriever.search(query, **kwargs)
    cache.set(query, response["results"])
    return {"results": response["results"], "from_cache": False}
```

```python
import redis, hashlib, json
import numpy as np

class SemanticCache:
    def __init__(self, redis_client, encoder, similarity_threshold=0.92, ttl_seconds=3600):
        self.redis = redis_client
        self.encoder = encoder
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

    def _query_id(self, embedding):
        return hashlib.sha256(embedding.astype(np.float32).tobytes()).hexdigest()[:16]

    def get(self, query):
        query_emb = self.encoder.encode(query)
        cached_ids = self.redis.smembers("cache:query_index")
        if not cached_ids:
            return None

        max_sim, best_id = -1.0, None
        for qid in cached_ids:
            qid = qid.decode('utf-8')
            cached_emb = json.loads(self.redis.get(f"cache:query:{qid}") or "null")
            if cached_emb is None:
                continue
            sim = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if sim > max_sim:
                max_sim, best_id = sim, qid

        if max_sim >= self.threshold and best_id:
            results = self.redis.get(f"cache:results:{best_id}")
            return json.loads(results) if results else None
        return None

    def set(self, query, results):
        emb = self.encoder.encode(query)
        qid = self._query_id(emb)
        self.redis.setex(f"cache:query:{qid}", self.ttl, json.dumps(emb.tolist()))
        self.redis.setex(f"cache:results:{qid}", self.ttl, json.dumps(results))
        self.redis.sadd("cache:query_index", qid)
```

**Threshold tuning**: 0.85-0.90 (50-70% hit rate, 85-92% precision — FAQ systems), 0.90-0.95 (40-60%, 92-98% — recommended default), 0.95-0.99 (20-40%, 98-99.5% — compliance-critical). For false positive management: seed with verified pairs, sample 1-5% of borderline hits for review, optionally validate with cross-encoder.

**Cache invalidation**: 1-hour TTL for dynamic content, 24-hour for stable, 7-day for reference. Add event-driven invalidation for critical accuracy. Alternatives: GPTCache, LangChain semantic cache, Redis Stack.

---

## Workflow: building a retrieval pipeline

Retrieval is a pipeline: query processing → stage 1 retrieval → stage 2 reranking → result validation. Start simple, add complexity as needed.

**Simple dense retrieval** (prototype, 20 lines):

```python
model = SentenceTransformer('all-mpnet-base-v2')
index = faiss.read_index("index.faiss")

def search(query: str, k: int = 5):
    query_embedding = model.encode(query).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k=k)
    return [{"chunk": chunks[i], "score": float(distances[0][j])} for j, i in enumerate(indices[0])]
```

**Two-stage pipeline** (production, +10-20% MRR):

```python
def two_stage_search(query: str, stage1_k: int = 50, final_k: int = 5):
    query_embedding = retriever.encode(query).reshape(1, -1).astype('float32')
    _, indices = index.search(query_embedding, k=stage1_k)
    stage1_chunks = [chunks[i] for i in indices[0]]

    rerank_scores = reranker.predict([[query, c] for c in stage1_chunks])
    return [stage1_chunks[i] for i in np.argsort(rerank_scores)[::-1][:final_k]]
```

**Hybrid with reranking** (production, handles diverse query types):

```python
class HybridRetriever:
    def search(self, query: str, stage1_k: int = 100, final_k: int = 5, alpha: float = 0.5):
        # Dense + sparse retrieval with score normalization
        # ... combine with alpha weighting ...
        # Rerank with cross-encoder
        # ... return top final_k ...
```

```python
class HybridRetriever:
    def __init__(self, index_path: str, chunks_path: str):
        self.retriever = SentenceTransformer('all-mpnet-base-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = faiss.read_index(index_path)
        self.chunks = np.load(chunks_path, allow_pickle=True)
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, stage1_k: int = 100, final_k: int = 5, alpha: float = 0.5):
        query_embedding = self.retriever.encode(query)
        dense_distances, dense_indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k=stage1_k)
        dense_scores = 1 / (1 + dense_distances[0])

        sparse_scores = self.bm25.get_scores(query.lower().split())

        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

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
from dataclasses import dataclass
from functools import lru_cache
import time, logging

@dataclass
class RetrievalMetrics:
    query: str
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms: float

class ProductionRetriever(HybridRetriever):
    def __init__(self, index_path, chunks_path):
        super().__init__(index_path, chunks_path)
        self.metrics_history = []

    @lru_cache(maxsize=1000)
    def _embed_query_cached(self, query: str):
        return tuple(self.retriever.encode(query).tolist())

    def search(self, query, stage1_k=100, final_k=5, alpha=0.5, enable_cache=True):
        start = time.time()
        # ... hybrid retrieval with timing per stage ...
        # ... error handling, metrics logging ...
        total_ms = (time.time() - start) * 1000
        logger.info(f"Retrieved {final_k} chunks in {total_ms:.1f}ms")
        return {"results": results, "metrics": {"total_latency_ms": total_ms}}

    def get_performance_summary(self):
        recent = self.metrics_history[-100:]
        return {
            "avg_total_latency_ms": np.mean([m.total_latency_ms for m in recent]),
            "p95_total_latency_ms": np.percentile([m.total_latency_ms for m in recent], 95),
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
| **Latency** | Predictable (200-500ms retrieval) | Variable (1-30s depending on iterations) |
| **Cost** | Low (1 LLM call for generation) | Higher (multiple LLM calls for reasoning) |
| **Best for** | Factual lookups, FAQ, single-hop questions | Multi-hop reasoning, research, complex analysis |
| **Predictability** | High (deterministic path) | Lower (agent may take unexpected paths) |

**When to use pipeline RAG**: Latency-sensitive applications (<500ms), predictable query types, FAQ/support systems, cost-sensitive workloads. **When to use agentic RAG**: Complex multi-hop questions, research tasks requiring synthesis across many sources, queries where the first retrieval may not contain the full answer.

### Key agentic patterns

**Tool-calling agent**: Give the LLM a `search_docs` tool. It queries, reads results, realizes it has a partial answer, formulates a new query for the missing part, and searches again. Implemented in LangGraph, LlamaIndex agents, and most agent frameworks.

```python
# Simplified agentic retrieval loop
tools = [{"name": "search_docs", "fn": lambda q: retriever.search(q, k=5)}]

def agentic_retrieve(question: str, max_iterations: int = 5):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_iterations):
        response = llm.invoke(messages, tools=tools)
        if response.tool_calls:
            for call in response.tool_calls:
                results = tools[0]["fn"](call.arguments["query"])
                messages.append({"role": "tool", "content": format_results(results)})
        else:
            return response.content  # Agent decided it has enough context
    return response.content
```

**Self-RAG** (Asai et al., 2023; ICLR 2024 oral, top 1%): The model generates special reflection tokens — `[Retrieve]` (should I retrieve?), `[ISREL]` (is the retrieved passage relevant?), `[ISSUP]` (is the generation supported?), `[ISUSE]` (is the response useful?) — enabling adaptive retrieval. The model only retrieves when it determines its knowledge is insufficient, and self-evaluates the quality of retrieved context before generating.

**Corrective RAG (CRAG)** (Yan et al., 2024): Adds a lightweight retrieval evaluator (fine-tuned T5-large) that classifies retrieved documents as Correct, Incorrect, or Ambiguous. On Correct, it applies a decompose-then-recompose algorithm to strip noise. On Incorrect, it discards retrieval and triggers web search as a fallback. On Ambiguous, it combines refined retrieval with web results. CRAG is plug-and-play — it layers on top of any existing RAG pipeline.

**Frameworks**: LangGraph (graph-based agent orchestration with explicit state), LlamaIndex (AgentRunner with built-in retrieval tools), Haystack (lower framework overhead than LangChain), CrewAI (multi-agent collaboration), and Autogen (conversational multi-agent) all support agentic RAG patterns.

**Practical recommendation**: Use pipeline RAG by default and trigger an agentic loop only when failure signals are detected — low retrieval confidence, missing citations, contradictory evidence, or user follow-ups indicating the initial answer was insufficient. This keeps most queries efficient while providing a recovery path for complex cases.

---

## Beyond text: emerging retrieval paradigms

Standard RAG retrieval assumes text-in, text-out. Two emerging paradigms extend retrieval beyond this boundary.

### GraphRAG (Knowledge Graph Retrieval)

Vector similarity retrieval struggles with queries requiring synthesis across many documents ("What are the main themes in this dataset?" or multi-hop reasoning "How does X relate to Y through Z?"). **GraphRAG** builds a knowledge graph from documents — extracting entities and relationships — then uses graph traversal (community detection, path finding) alongside vector similarity.

Microsoft's GraphRAG approach (2024): (1) extract entities and relationships from chunks using an LLM, (2) build a graph and detect communities via hierarchical Leiden algorithm, (3) generate community summaries at multiple granularity levels, (4) at query time, retrieve via both vector search on chunks and graph traversal on communities. In Microsoft's evaluation, GraphRAG achieved ~80% accuracy on global sensemaking questions vs ~50% for standard vector RAG, with 72-83% comprehensiveness scores. The trade-off is significantly higher indexing cost (many LLM calls for extraction + summarization), though LazyGraphRAG (2025) reduces this to 0.1% of the cost.

**When GraphRAG helps**: Multi-hop reasoning, thematic/summarization queries across large corpora, datasets with rich entity relationships. **When to skip**: Simple factual lookups, small document sets, latency-constrained systems (graph construction is expensive).

### Multimodal retrieval

Text-only RAG fails for documents with charts, diagrams, tables, or images. Two approaches:

**CLIP-based retrieval**: Encode images and text into a shared embedding space. Query with text, retrieve relevant images (or vice versa). Works for image search but does not understand document layout.

**ColPali and vision-language models** (ICLR 2025): Encode entire document pages as images using a vision-language model (PaliGemma), skipping OCR and parsing entirely. ColPali uses a late-interaction architecture (like ColBERT) over ~1,030 page image patches, achieving strong retrieval on visually-rich documents like PDFs with tables, figures, and complex layouts. Successors include ColQwen2 (multi-resolution) and domain-specific variants. Trade-off: dramatically simpler indexing pipeline vs ~100x more vectors per page than single-vector models.

**When multimodal matters**: Document collections with significant visual content (scientific papers, financial reports, slide decks, manuals with diagrams). **When to skip**: Text-heavy corpora where standard chunking and embedding works well.

---

## Common pitfalls

**Using only dense retrieval**: Misses exact keyword matches. "HIPAA compliance" should strongly favor documents containing "HIPAA." Use hybrid search.

**Skipping reranking**: If you have >50ms latency budget, add reranking. Typically improves MRR by 10-20%.

**Reranking too few candidates**: Retrieve 50-100 in stage 1, rerank to 10-20, return top 5. Give the reranker room to correct stage 1 errors.

**Not tuning efSearch**: HNSW defaults (efSearch=16) prioritize speed over recall. Set efSearch=50-100 for 95-98% recall.

**Returning low-scoring results**: If all top-5 scores are <0.3, return "no relevant information" instead of showing irrelevant chunks.

**Ignoring query length**: Single-word queries need expansion. 50+ word queries need decomposition.

**Not caching embeddings**: If 20% of queries repeat, cache their embeddings (30ms → <1ms).

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

**Using ColBERT for <100 candidates**: Cross-encoder is fast enough and more accurate at that scale.

**Over-aggressive cache threshold (0.95-0.99)**: Barely better than exact-match. Use 0.90-0.95 for 40-60% hit rate.

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
# Ragas v0.2+ synthetic test generation (KnowledgeGraph-based)
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper

generator = TestsetGenerator(llm=LangchainLLMWrapper(llm))
testset = generator.generate_with_langchain_docs(
    documents, testset_size=200,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
# Returns DataFrame with: question, ground_truth, contexts, evolution_type

# LlamaIndex equivalent
from llama_index.core.evaluation import generate_question_context_pairs

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)
# Returns EmbeddingQAFinetuneDataset with queries, corpus, relevance mappings
```

**Automated evaluation frameworks**:

| Framework | Key metrics | Unique strength | Cost |
|-----------|------------|-----------------|------|
| **Ragas** | Context precision, context recall, faithfulness, answer relevancy | End-to-end RAG eval + synthetic test generation | LLM calls (~$0.50-2/run for 200 queries) |
| **TruLens** | Groundedness, relevance, sentiment | Real-time dashboard, production monitoring | LLM calls (similar) |
| **ARES** (Stanford) | Prediction-powered inference | Statistically rigorous confidence intervals from fewer labels | Minimal LLM calls |
| **DeepEval** | 14+ metrics, G-Eval | CI/CD native (pytest plugin), unit test syntax | LLM calls |

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

| Use Case | Hit Rate@5 | Latency | Cost/1K Queries |
|----------|-----------|---------|-----------------|
| FAQ chatbot | >85% | <100ms p50 | $0.05-0.10 |
| Enterprise search | >90% | <150ms p50 | $0.10-0.20 |
| Legal/Medical | >95% | <200ms p50 | $0.20-0.50 |
| Internal KB | >80% | <200ms p50 | $0.02-0.05 |

**Cost optimization**: semantic caching (40-70% cost reduction), skip reranking for simple queries, hybrid search over dense-only (free quality gain), batch embeddings.

---

## Rollout & change management

Phased deployment strategy to minimize risk and measure impact: internal beta → limited production → full rollout → deprecate baseline.

### Phased rollout strategy

| Phase | Duration | Audience | Success Criteria |
|-------|----------|----------|-----------------|
| **1. Internal beta** | 2-4 weeks | 10-20 internal users | >80% correct, Hit Rate@5 >85%, p99 <200ms |
| **2. Limited production** | 4-6 weeks | 10-25% users (A/B test) | Rating >3.8/5.0 AND >10% over baseline |
| **3. Full rollout** | 2-4 weeks | 25% → 50% → 75% → 100% | Stable at scale, p95 <200ms, error <2% |
| **4. Deprecate baseline** | 30 days | Shadow mode only | Keep for emergency rollback, then decommission |

### A/B testing framework

- Split by user ID hash (consistent per-user experience)
- Minimum 1000 queries per group, recommended 5000+ for 2-3% effect detection
- Compare: user rating, task completion, p95 latency, cost per 1K queries

### Rollback plan

**Triggers**: Hit Rate@5 <75%, user satisfaction <3.0/5.0 for >30%, p99 >1000ms for >5%, error rate >10%.
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
| **Index corruption** | Wrong results, user satisfaction drops | Daily snapshots, automated chunk count validation, quarterly DR drill |

### Quality risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Quality degradation over time** | Hit Rate drifts as documents, users, or query patterns change | Weekly automated eval, quarterly embedding retraining, document freshness monitoring, user feedback loop |
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
- **On-call**: 3-5 engineers rotating weekly, expect 1-2 incidents per quarter

---

## Best practices

**Default to two-stage retrieval.** Stage 1 retrieves 50-100 candidates in <50ms, stage 2 reranks to 5-10 in <50ms. MRR improvement 10-20%. This covers the majority of use cases. For enterprise systems with strict accuracy and latency SLAs (>300ms budget), consider three-stage retrieval (two-tower → ColBERT → cross-encoder) for an additional 2-5% accuracy gain.

**Default to hybrid search.** Dense + sparse with alpha=0.5. Adjust per query type.

**Optimize stage 1 for recall, stage 2 for precision.** Use efSearch=50-100 for HNSW. It's okay if stage 1 ranks the correct chunk 20th as long as it's in the top 50.

**Cache query embeddings for common queries.** 20% repeat rate → 30ms to <1ms.

**Monitor retrieval independently of generation.** Track Hit Rate@5 and MRR separately. Diagnose whether failures are retrieval or generation problems.

**Use metadata filtering for explicit constraints only.** Don't filter by relevance score — let reranking handle that.

**Rerank at least 2x more candidates than you return.** LLM needs 5 chunks → retrieve 100, rerank to 50, return top 5.

**Profile latency at each stage.** Measure stage 1, stage 2, and total separately.

**Use query decomposition for multi-hop questions.** "Compare X and Y" needs separate retrievals.

**A/B test retrieval strategies.** If two-stage improves MRR by <5%, the complexity isn't justified.

**Apply MMR for multi-faceted queries, skip for precision search.** Redundancy confirms correctness in factual lookups.

**Apply compression after reranking, not before.** Only compress high-quality reranked candidates.

**Choose parent-child strategy by document structure.** Linear → sentence window (size 3-5). Hierarchical → auto-merging (threshold 0.6).

**Implement semantic caching for FAQ/support systems.** 40-70% hit rate, 4-8x faster, 40-73% cost savings.

**Use Matryoshka embeddings for large-scale indexes.** Truncate to 256-dim for stage 1 with ~2-4% accuracy drop; let the cross-encoder recover precision in stage 2.

**Add self-querying when users specify constraints.** Dates, categories, and named entities in queries should become metadata filters, not embedding noise.

**Default to pipeline RAG; move to agentic when pipeline fails.** Pipeline RAG handles 80%+ of use cases at 200-500ms. Reserve agentic RAG for multi-hop reasoning and complex research queries.

**Automate retrieval evaluation with synthetic test data.** Generate 200-500 question-context pairs from your corpus, run MRR@10 and Hit Rate@5 on every deployment, set regression thresholds.
