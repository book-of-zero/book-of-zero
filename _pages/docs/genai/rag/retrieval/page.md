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
  - [Why routing matters](#why-routing-matters)
  - [Intent classification](#intent-classification)
  - [Routing logic](#routing-logic)
  - [When routing matters](#when-routing-matters)
  - [Advanced: Multi-step routing](#advanced-multi-step-routing)
- [Stage 1: Fast retrieval](#stage-1-fast-retrieval)
  - [Two-tower architecture](#two-tower-architecture)
  - [BM25 sparse retrieval](#bm25-sparse-retrieval)
  - [Hybrid search](#hybrid-search)
- [Stage 2: Reranking](#stage-2-reranking)
  - [ColBERT (Late Interaction)](#colbert-late-interaction)
- [Search-time index tuning](#search-time-index-tuning)
- [Query optimization](#query-optimization)
  - [Query expansion](#query-expansion)
  - [Conversational query rewriting](#conversational-query-rewriting)
  - [Query decomposition](#query-decomposition)
  - [Metadata filtering and boosting](#metadata-filtering-and-boosting)
    - [Hard security filtering (RBAC)](#hard-security-filtering-rbac)
  - [Advanced query techniques](#advanced-query-techniques)
- [Result optimization](#result-optimization)
  - [MMR (Maximal Marginal Relevance)](#mmr-maximal-marginal-relevance)
  - [Contextual compression](#contextual-compression)
  - [Parent-child retrieval](#parent-child-retrieval)
  - [Document-level vs chunk-level retrieval](#document-level-vs-chunk-level-retrieval)
- [Workflow: building a retrieval pipeline](#workflow-building-a-retrieval-pipeline)
  - [Simple dense retrieval](#simple-dense-retrieval)
  - [Two-stage pipeline](#two-stage-pipeline)
  - [Hybrid with reranking](#hybrid-with-reranking)
  - [Production pipeline with monitoring](#production-pipeline-with-monitoring)
    - [Semantic caching](#semantic-caching)
- [Common pitfalls](#common-pitfalls)
- [Production quality metrics & SLAs](#production-quality-metrics--slas)
  - [Retrieval quality benchmarks](#retrieval-quality-benchmarks)
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

**Two-tower architecture**: A retrieval architecture where queries and documents are encoded independently by the same model, producing vectors that can be compared via cosine similarity. Enables precomputation — all document embeddings are computed offline, only query embedding happens at search time.

**Dense retrieval**: Semantic search using learned embeddings. Represents queries and documents as dense vectors. Captures semantic similarity but misses exact keyword matches. DPR (Dense Passage Retrieval) is the landmark implementation.

**Sparse retrieval**: Keyword-based search using term frequencies. Represents documents as sparse vectors (most dimensions are zero). BM25 is the standard algorithm. Fast and interpretable but fails on semantic paraphrases.

**Hybrid search**: Combining dense and sparse retrieval. Retrieves candidates from both systems, merges results with score normalization, then reranks. Typically improves Hit Rate@10 by 5-15% over either method alone.

**Cross-encoder**: A reranking model that jointly encodes query and document pairs. More accurate than two-tower (sees full interaction) but 100× slower (cannot precompute). Used in stage 2 to refine top-k candidates from stage 1.

**ColBERT (Contextualized Late Interaction over BERT)**: A late interaction architecture that encodes queries and documents as sequences of token-level embeddings (N tokens × 128-dim) rather than single sequence-level vectors. Similarity computed via MaxSim operation at query time. Bridges the gap between two-tower speed and cross-encoder accuracy — 95-98% of cross-encoder accuracy at 10× lower latency.

**ANN (Approximate Nearest Neighbor)**: Algorithms that trade exact accuracy for speed. Find the approximate top-k nearest neighbors in sublinear time. HNSW, IVF, and PQ are common implementations. Production RAG requires ANN — exact search doesn't scale past 10K documents.

**RBAC (Role-Based Access Control)**: Security filtering that physically blocks unauthorized chunks from retrieval results based on user roles, departments, or permissions. Implemented via metadata filtering at query time (user token → vector DB filter → allowlist-based retrieval). Distinct from soft boosting which adjusts scores but doesn't block access.

**Semantic caching**: Caching retrieval results based on query similarity (cosine similarity ≥ 0.90-0.95 threshold) rather than exact string matching. Handles paraphrased queries ("What are HIPAA rules?" vs "Explain HIPAA requirements") and achieves 40-70% hit rates (vs 10-20% for exact-match) in FAQ/support systems.

---

## Query routing & intent classification

Not all queries should go to the vector database. Some need SQL, some are greetings, some are commands. **Query routing** (also called semantic routing) classifies query intent and routes to the appropriate backend. Skip this step and your RAG system will waste time retrieving documents for "Hello" or "What's my account balance?" (which needs a database query, not document search).

### Why routing matters

**Problem**: Your RAG system is a customer support chatbot with access to:
- Knowledge base (vector DB): FAQ articles, product docs, troubleshooting guides
- User database (SQL): Account info, order history, subscription status
- Weather API (external): Current weather, forecasts
- Direct responses (no retrieval): Greetings, small talk, out-of-scope

**Without routing**, every query hits the vector DB:
- User: "Hello" → Vector DB returns random chunks about greetings in docs → LLM responds awkwardly
- User: "What's my account balance?" → Vector DB returns chunks about account types → LLM can't answer (data not in docs)
- User: "What's the weather in SF?" → Vector DB returns irrelevant chunks → LLM hallucinates or says "I don't know"

**With routing**, queries go to the right backend:
- "Hello" → Direct response (no retrieval): "Hi! How can I help you today?"
- "What's my account balance?" → SQL query: `SELECT balance FROM accounts WHERE user_id = ...` → "Your account balance is $127.43"
- "What's the weather in SF?" → Weather API → "Currently 62°F and sunny in San Francisco"
- "How do I reset my password?" → Vector DB (knowledge base) → Returns relevant FAQ article

### Intent classification

Classify queries into predefined intents, each mapped to a backend.

#### Intent categories

| Intent | Description | Backend | Example Queries |
|--------|-------------|---------|-----------------|
| **Greeting** | Small talk, greetings, thanks | Direct response (no retrieval) | "Hi", "Hello", "Thanks", "Goodbye" |
| **Knowledge base** | Questions answerable from docs | Vector DB (RAG) | "How do I reset password?", "What is RAG?", "Explain HIPAA" |
| **Structured data** | Questions about user data, transactions | SQL database | "What's my balance?", "Show my orders", "When does my subscription expire?" |
| **Real-time data** | Live info not in knowledge base | External API (weather, stock prices, news) | "What's the weather?", "AAPL stock price", "Latest news" |
| **Out of scope** | Questions outside system capabilities | Direct response (apologize) | "What's the meaning of life?", "Tell me a joke", "Write code for me" |

#### Simple implementation (rule-based)

For small systems with <10 intents, use keyword matching:

```python
import re
from typing import Literal

IntentType = Literal["greeting", "knowledge_base", "structured_data", "real_time", "out_of_scope"]

def classify_intent_simple(query: str) -> IntentType:
    """
    Rule-based intent classification using keyword matching.

    Fast (no ML), deterministic, but limited to simple patterns.
    """
    query_lower = query.lower()

    # Greeting patterns
    if re.search(r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b', query_lower):
        return "greeting"

    # Structured data patterns (account, balance, order, subscription)
    if re.search(r'\b(my account|my balance|my order|my subscription|purchase history)\b', query_lower):
        return "structured_data"

    # Real-time data patterns (weather, stock, news)
    if re.search(r'\b(weather|temperature|forecast|stock price|news)\b', query_lower):
        return "real_time"

    # Out-of-scope patterns (jokes, creative writing, coding)
    if re.search(r'\b(joke|poem|story|write code|meaning of life)\b', query_lower):
        return "out_of_scope"

    # Default: Knowledge base (most queries)
    return "knowledge_base"

# Examples
classify_intent_simple("Hi there!")                    # "greeting"
classify_intent_simple("How do I reset my password?")  # "knowledge_base"
classify_intent_simple("What's my account balance?")   # "structured_data"
classify_intent_simple("Weather in SF?")               # "real_time"
classify_intent_simple("Tell me a joke")               # "out_of_scope"
```

#### Production implementation (ML-based)

For complex systems with 10+ intents or nuanced queries, use a classifier:

```python
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Train classifier on labeled examples

# Training data: (query, intent) pairs
training_data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("How are you?", "greeting"),
    ("How do I reset my password?", "knowledge_base"),
    ("What is RAG?", "knowledge_base"),
    ("Explain HIPAA compliance", "knowledge_base"),
    ("What's my account balance?", "structured_data"),
    ("Show my recent orders", "structured_data"),
    ("When does my subscription expire?", "structured_data"),
    ("What's the weather in SF?", "real_time"),
    ("AAPL stock price", "real_time"),
    ("Tell me a joke", "out_of_scope"),
    ("What's the meaning of life?", "out_of_scope"),
    # ... 100-1000+ examples
]

queries, intents = zip(*training_data)

# 2. Embed queries
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode(queries)

# 3. Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(query_embeddings, intents)

# 4. Classify new queries
def classify_intent_ml(query: str) -> IntentType:
    """ML-based intent classification (more accurate than rules)."""
    query_embedding = model.encode([query])[0]
    intent = classifier.predict([query_embedding])[0]
    return intent

# Examples (same as above, but handles variations better)
classify_intent_ml("hey there!")                       # "greeting" (trained on "Hi", "Hello")
classify_intent_ml("password reset instructions?")     # "knowledge_base" (semantic similarity)
classify_intent_ml("how much money do I have?")        # "structured_data" (paraphrase of "balance")
```

#### Using embeddings for routing (semantic similarity)

Instead of training a classifier, use few-shot semantic similarity:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define example queries for each intent
intent_examples = {
    "greeting": [
        "Hi", "Hello", "Hey there", "Good morning", "Thanks", "Goodbye"
    ],
    "knowledge_base": [
        "How do I reset my password?",
        "What is RAG?",
        "Explain HIPAA compliance",
        "How does two-factor authentication work?"
    ],
    "structured_data": [
        "What's my account balance?",
        "Show my recent orders",
        "When does my subscription expire?",
        "What's my billing address?"
    ],
    "real_time": [
        "What's the weather in San Francisco?",
        "AAPL stock price",
        "Latest news about AI",
        "Current time in Tokyo"
    ],
    "out_of_scope": [
        "Tell me a joke",
        "What's the meaning of life?",
        "Write me a poem",
        "Solve this math problem"
    ]
}

# Precompute embeddings for example queries
intent_embeddings = {
    intent: model.encode(examples)
    for intent, examples in intent_examples.items()
}

def classify_intent_semantic(query: str, threshold: float = 0.5) -> IntentType:
    """
    Classify intent by semantic similarity to example queries.

    Args:
        query: User query
        threshold: Minimum similarity to match intent (0-1). Lower = more aggressive routing.

    Returns:
        Intent with highest similarity, or "knowledge_base" if no match above threshold.
    """
    query_embedding = model.encode(query)

    # Compute similarity to each intent's examples
    intent_scores = {}
    for intent, embeddings in intent_embeddings.items():
        # Max similarity across all examples for this intent
        similarities = util.cos_sim(query_embedding, embeddings)[0]
        intent_scores[intent] = float(similarities.max())

    # Get best matching intent
    best_intent = max(intent_scores, key=intent_scores.get)
    best_score = intent_scores[best_intent]

    # If score too low, default to knowledge_base
    if best_score < threshold:
        return "knowledge_base"

    return best_intent

# Examples
classify_intent_semantic("hey!")                        # "greeting" (0.85 similarity)
classify_intent_semantic("how to reset password")       # "knowledge_base" (0.78 similarity)
classify_intent_semantic("my account balance please")   # "structured_data" (0.72 similarity)
classify_intent_semantic("weather today")               # "real_time" (0.68 similarity)
classify_intent_semantic("some random unrelated query") # "knowledge_base" (default, <0.5 similarity)
```

### Routing logic

After classifying intent, route to the appropriate backend:

```python
from typing import Dict, Any

def route_query(query: str, user_id: str) -> Dict[str, Any]:
    """
    Route query to appropriate backend based on intent.

    Returns:
        {"intent": str, "response": str, "source": str}
    """
    intent = classify_intent_semantic(query)

    if intent == "greeting":
        # Direct response (no retrieval)
        return {
            "intent": "greeting",
            "response": "Hi! How can I help you today?",
            "source": "direct"
        }

    elif intent == "knowledge_base":
        # Vector DB (RAG retrieval)
        chunks = hybrid_search(query, documents, k=5)
        response = llm_generate(query, chunks)  # LLM generates answer from retrieved chunks
        return {
            "intent": "knowledge_base",
            "response": response,
            "source": "vector_db"
        }

    elif intent == "structured_data":
        # SQL query
        sql_query = generate_sql(query, user_id)  # Use text-to-SQL model
        result = database.execute(sql_query)
        response = format_sql_result(result)  # "Your account balance is $127.43"
        return {
            "intent": "structured_data",
            "response": response,
            "source": "sql"
        }

    elif intent == "real_time":
        # External API
        if "weather" in query.lower():
            weather_data = call_weather_api(query)
            response = format_weather(weather_data)
        elif "stock" in query.lower():
            stock_data = call_stock_api(query)
            response = format_stock(stock_data)
        else:
            response = "I can help with weather and stock prices. What would you like to know?"

        return {
            "intent": "real_time",
            "response": response,
            "source": "api"
        }

    elif intent == "out_of_scope":
        # Polite refusal
        return {
            "intent": "out_of_scope",
            "response": "I'm designed to help with account questions and product support. For other topics, I recommend searching online or asking a general-purpose assistant.",
            "source": "direct"
        }

    else:
        # Fallback (shouldn't happen)
        return {
            "intent": "unknown",
            "response": "I'm not sure how to help with that. Could you rephrase your question?",
            "source": "direct"
        }

# Usage
result = route_query("How do I reset my password?", user_id="user_123")
# {"intent": "knowledge_base", "response": "To reset your password, go to...", "source": "vector_db"}

result = route_query("What's my account balance?", user_id="user_123")
# {"intent": "structured_data", "response": "Your account balance is $127.43", "source": "sql"}
```

### When routing matters

**Critical for**:
- Multi-backend systems (RAG + SQL + APIs)
- Customer support chatbots (mix of FAQ + account queries + greetings)
- Enterprise assistants (RAG + CRM + calendar + email)

**Less important for**:
- Single-backend systems (pure knowledge base search)
- Specialized tools (medical Q&A over clinical papers only)

**Cost-benefit**:
- **Cost**: 5-20ms latency (semantic similarity), <1ms (rule-based), 0.5-1 week engineering time
- **Benefit**: Avoid 30-60% of useless retrieval (greetings, out-of-scope, structured data), faster responses (direct answers vs retrieval + generation), better UX (correct backend = correct answer)

### Advanced: Multi-step routing

For complex queries that need multiple backends, use agentic routing:

```python
def route_complex(query: str) -> list[Dict[str, Any]]:
    """
    Route complex queries to multiple backends.

    Example: "Show my recent orders and recommend similar products"
    → Route 1: SQL (fetch recent orders)
    → Route 2: Vector DB (retrieve product recommendations based on order history)
    """
    # Use LLM to decompose query into steps
    steps = decompose_query(query)  # LLM breaks into: ["Show my recent orders", "Recommend similar products"]

    results = []
    for step in steps:
        intent = classify_intent_semantic(step)
        result = route_query(step, user_id="user_123")
        results.append(result)

    # Combine results
    combined_response = combine_responses(results)
    return combined_response
```

---

## Stage 1: Fast retrieval

Stage 1 retrieves 50-100 candidate chunks from millions in <50ms. The goal is high recall — you want the correct chunk in the top 50 even if it's not ranked 1st. Precision comes in stage 2.

### Two-tower architecture

The two-tower architecture is why dense retrieval scales. You encode queries and documents separately with the same encoder, then compare via cosine similarity.

```
┌─────────────┐          ┌─────────────┐
│   Query     │          │  Document   │
│  "What is   │          │  "RAG stands│
│   RAG?"     │          │   for..."   │
└──────┬──────┘          └──────┬──────┘
       │                        │
       │  Encoder (shared)      │  Encoder (shared)
       │                        │
       ▼                        ▼
  [ 0.2, 0.8, ... ]        [ 0.3, 0.7, ... ]
       Query vector            Doc vector
              │                    │
              └────────┬───────────┘
                       ▼
                Cosine similarity
                     0.92
```

**Why this works**: Documents are encoded once at index time. At search time, you only encode the query (5-50ms) then compute cosine similarity against precomputed document vectors (<1ms per document with ANN). Compare to a cross-encoder that must re-encode every query-document pair — that's 50-100ms per document.

**Implementation with Sentence Transformers**:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load two-tower model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Index time: encode all documents (once)
documents = ["RAG stands for Retrieval-Augmented Generation...", "Paris is the capital...", ...]
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Build FAISS index for fast similarity search
dimension = doc_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(doc_embeddings)

# Search time: encode query, find nearest neighbors
query = "What is RAG?"
query_embedding = model.encode(query, convert_to_numpy=True)

# Retrieve top 50 candidates (<50ms)
distances, indices = index.search(query_embedding[np.newaxis, :], k=50)

# indices contains the document IDs of the top 50 results
top_docs = [documents[i] for i in indices[0]]
```

**DPR (Dense Passage Retrieval)**: The landmark two-tower architecture from Facebook AI Research. Uses BERT-based encoders fine-tuned on question-passage pairs. Trained with contrastive loss: positive pairs (question, correct passage) should have high similarity, negative pairs (question, irrelevant passage) should have low similarity.

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch

# Load DPR models
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
c_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Encode documents (index time)
doc_tokens = c_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    doc_embeddings = c_encoder(**doc_tokens).pooler_output.numpy()

# Encode query (search time)
query_tokens = q_tokenizer([query], return_tensors="pt")
with torch.no_grad():
    query_embedding = q_encoder(**query_tokens).pooler_output.numpy()

# Compute similarities
similarities = np.dot(doc_embeddings, query_embedding.T).squeeze()
top_indices = np.argsort(similarities)[::-1][:50]
```

**When two-tower works**: Queries are semantic paraphrases of document content. "What causes rain?" matches "Precipitation occurs when atmospheric water vapor condenses..." even though they share no keywords.

**When two-tower fails**: Exact keyword matches matter. "HIPAA compliance requirements" should strongly match documents containing "HIPAA" even if the semantic similarity is moderate. Solution: hybrid search (next section).

### BM25 sparse retrieval

BM25 (Best Matching 25) is the standard keyword-based ranking algorithm. It scores documents by term frequency (TF) and inverse document frequency (IDF), with saturation to prevent over-weighting of repeated terms.

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Search
query = "HIPAA compliance requirements"
tokenized_query = query.lower().split()

# Get top 50 candidates
scores = bm25.get_scores(tokenized_query)
top_indices = np.argsort(scores)[::-1][:50]
top_docs = [documents[i] for i in top_indices]
```

**How BM25 works**:
- **Term frequency (TF)**: More occurrences of query terms = higher score. But with saturation — the 10th occurrence of "HIPAA" adds less value than the 1st.
- **Inverse document frequency (IDF)**: Rare terms are more valuable. "HIPAA" is more discriminative than "the".
- **Document length normalization**: Long documents aren't unfairly penalized, but repetition in short documents is rewarded.

**Formula** (simplified):
```
score(Q, D) = Σ IDF(q_i) × (TF(q_i, D) × (k1 + 1)) / (TF(q_i, D) + k1 × (1 - b + b × |D| / avgDL))
```
where k1 (term saturation, default 1.5) and b (length normalization, default 0.75) are tuning parameters.

**When BM25 works**: Exact keyword matches, named entities, acronyms. "Find documents mentioning 'GPT-4'" — BM25 will rank documents with "GPT-4" at the top.

**When BM25 fails**: Semantic paraphrases. "What causes rain?" won't match "Precipitation occurs..." because they share no keywords.

### Hybrid search

Combine dense and sparse retrieval to get the best of both: semantic understanding and keyword precision.

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')
bm25 = BM25Okapi(tokenized_docs)

def hybrid_search(query: str, documents: list[str], k: int = 50, alpha: float = 0.5):
    """
    Hybrid search combining dense and sparse retrieval.

    Args:
        query: Search query
        documents: List of documents
        k: Number of results to return
        alpha: Weight for dense retrieval (1-alpha for sparse). 0.5 = equal weight.

    Returns:
        List of (doc_index, score) tuples
    """
    # Dense retrieval
    query_embedding = model.encode(query, convert_to_numpy=True)
    doc_embeddings = model.encode(documents, convert_to_numpy=True)
    dense_scores = np.dot(doc_embeddings, query_embedding)

    # Sparse retrieval
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)

    # Normalize scores to [0, 1] for fair combination
    dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
    sparse_scores_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

    # Combine scores
    hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * sparse_scores_norm

    # Get top-k
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    results = [(i, hybrid_scores[i]) for i in top_indices]

    return results

# Search with hybrid
results = hybrid_search("HIPAA compliance requirements", documents, k=50, alpha=0.5)
top_docs = [documents[i] for i, score in results]
```

**Score normalization is critical**: Dense scores might range [0.3, 0.9] while BM25 scores range [0, 150]. Without normalization, BM25 dominates. Min-max normalization maps both to [0, 1].

**Tuning alpha**:
- alpha=1.0: pure dense retrieval (semantic, misses exact matches)
- alpha=0.0: pure sparse retrieval (keyword, misses paraphrases)
- alpha=0.5: balanced (default starting point)
- alpha=0.7: favor semantic (good for conversational queries)
- alpha=0.3: favor keywords (good for technical/entity search)

**Empirical results**: Hybrid search typically improves Hit Rate@10 by 5-15% over dense-only or sparse-only. The gain is largest when queries mix semantic and keyword requirements ("Explain HIPAA's privacy rules" — needs both "HIPAA" keyword match and semantic understanding of "privacy rules").

**Alternative fusion: Reciprocal Rank Fusion (RRF)**: Instead of score combination, merge ranked lists.

```python
def reciprocal_rank_fusion(dense_ranks: list[int], sparse_ranks: list[int], k: int = 60):
    """
    RRF: score(doc) = Σ 1 / (k + rank(doc))

    Combines ranked lists without requiring score normalization.
    k=60 is a standard default from the original paper.
    """
    scores = {}

    # Add scores from dense retrieval
    for rank, doc_id in enumerate(dense_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Add scores from sparse retrieval
    for rank, doc_id in enumerate(sparse_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# Get ranked lists from both retrievers
dense_top_indices = np.argsort(dense_scores)[::-1][:100].tolist()
sparse_top_indices = np.argsort(sparse_scores)[::-1][:100].tolist()

# Fuse rankings
fused_results = reciprocal_rank_fusion(dense_top_indices, sparse_top_indices)
top_docs = [documents[doc_id] for doc_id, score in fused_results[:50]]
```

**RRF vs score combination**: RRF is more robust to different score scales but less tunable (no alpha parameter). Use RRF when you don't have time to tune alpha. Use score combination when you need control over the dense/sparse balance.

---

## Stage 2: Reranking

Stage 1 gave you 50 candidates. Stage 2 refines them to the top 5-10 using a more accurate but slower model.

### Cross-encoder architecture

Unlike two-tower (separate encoders), cross-encoders jointly encode query and document. This captures interaction — the model sees "does THIS query match THIS document?" not just "are these embeddings similar?"

```
Two-tower (stage 1):             Cross-encoder (stage 2):
┌─────────┐  ┌─────────┐         ┌───────────────────────┐
│  Query  │  │Document │         │ [CLS] Query [SEP]     │
└────┬────┘  └────┬────┘         │      Document [SEP]   │
     │            │               └───────────┬───────────┘
Encoder       Encoder                         │
     │            │                       Encoder
     ▼            ▼                            │
 [0.2, ...]  [0.3, ...]                       ▼
     │            │                        [0.2, ...]
     └─────┬──────┘                            │
           ▼                               Linear
    Cosine sim                                 │
        0.92                                   ▼
                                          Relevance score
                                              0.94
```

**Why cross-encoders are more accurate**: They see attention between query and document tokens. If the query is "capital of France" and the document says "Paris is the capital of France", the cross-encoder's attention mechanism directly connects "capital" in the query to "capital" in the document and "Paris" nearby. The two-tower only sees "query embedding is close to document embedding" without token-level interaction.

**Why cross-encoders are slower**: You must encode every query-document pair. For 50 candidates, that's 50 forward passes (2-5 seconds on GPU). Two-tower needs 1 query encoding + 50 similarity computations (<50ms).

**Implementation**:

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder (trained for passage reranking)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Stage 1: Get 50 candidates from two-tower or hybrid search
candidates = hybrid_search(query, documents, k=50)
candidate_docs = [documents[i] for i, score in candidates]

# Stage 2: Rerank with cross-encoder
pairs = [[query, doc] for doc in candidate_docs]
rerank_scores = reranker.predict(pairs)  # 50-100ms per candidate on GPU

# Sort by reranking score
reranked_indices = np.argsort(rerank_scores)[::-1]
top_5_docs = [candidate_docs[i] for i in reranked_indices[:5]]
```

**Models**:
- **ms-marco-MiniLM-L-6-v2**: Fast (6 layers), good for <100 candidates. Default choice.
- **ms-marco-MiniLM-L-12-v2**: Better quality (12 layers) but 2× slower.
- **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1**: Multilingual (100+ languages).

### When reranking matters

**Empirical improvement**: Reranking typically improves MRR (Mean Reciprocal Rank) by 10-20% and Hit Rate@5 by 5-15% over stage 1 alone. The gain is largest when:
- Stage 1 has high recall but low precision (correct chunk is in top 50, but ranked 20th)
- Queries require nuanced understanding ("compare X and Y" needs to find chunks that discuss both X and Y, not separate chunks)

**When to skip reranking**:
- Latency budget is tight (<100ms total retrieval time)
- Stage 1 already achieves >95% Hit Rate@5 (diminishing returns)
- You're retrieving 20+ chunks for the LLM (precision matters less)

**Production pattern**: Two-stage pipeline with 100 candidates → rerank to 50 → LLM uses top 5-10.

```python
def two_stage_retrieval(query: str, documents: list[str], stage1_k: int = 100, stage2_k: int = 50, final_k: int = 5):
    """
    Two-stage retrieval: fast stage 1 (100 candidates) → accurate stage 2 (rerank to 50) → top-k for LLM.
    """
    # Stage 1: Hybrid search (dense + sparse)
    stage1_results = hybrid_search(query, documents, k=stage1_k, alpha=0.5)
    stage1_docs = [documents[i] for i, score in stage1_results]

    # Stage 2: Cross-encoder reranking
    pairs = [[query, doc] for doc in stage1_docs]
    rerank_scores = reranker.predict(pairs)

    # Get top stage2_k from reranking
    stage2_indices = np.argsort(rerank_scores)[::-1][:stage2_k]

    # Final top-k for LLM
    final_indices = stage2_indices[:final_k]
    final_docs = [stage1_docs[i] for i in final_indices]

    return final_docs

# Retrieve top 5 chunks for LLM
top_chunks = two_stage_retrieval("What are HIPAA's privacy rules?", documents, stage1_k=100, stage2_k=50, final_k=5)
```

**Latency breakdown** (typical GPU inference):
- Stage 1 hybrid search (100 candidates): 30-50ms
- Stage 2 reranking (100 → 50): 30-50ms
- **Total**: 60-100ms

This fits within the 100ms retrieval budget for interactive RAG systems.

### ColBERT (Late Interaction)

Cross-encoders are too slow for reranking 100-1000 candidates (50-500ms). Two-tower models are fast but less accurate. **ColBERT** (Contextualized Late Interaction over BERT) is the middle ground: token-level embeddings with late interaction (MaxSim operation) that approaches cross-encoder accuracy at near two-tower speed.

**Problem**: You have 100-1000 candidates from stage 1 (two-tower retrieval). Cross-encoder reranking would take 50-500ms (too slow). Two-tower models are fast but accuracy is limited. You need a faster alternative that's more accurate than two-tower but faster than cross-encoder.

**Architecture**: ColBERT encodes queries and documents independently (like two-tower) but at **token-level granularity** rather than sequence-level. Similarity is computed via **late interaction** (MaxSim operation) at query time rather than during encoding.

```
Two-tower (bi-encoder):
  Query → [single 768-dim vector]
  Document → [single 768-dim vector]
  Similarity = dot product (fast, lower accuracy)

Cross-encoder:
  [Query, Document] → BERT → scalar score
  Similarity = forward pass through transformer (slow, high accuracy)

ColBERT (late interaction):
  Query → [N query tokens × 128-dim vectors]  (e.g., 10 tokens × 128-dim)
  Document → [M doc tokens × 128-dim vectors]  (e.g., 200 tokens × 128-dim)
  Similarity = MaxSim(query_tokens, doc_tokens)  (fast, high accuracy)

MaxSim operation (per query token, find max similarity with any doc token):
  For each query token Qi:
    score_i = max(cosine(Qi, Dj) for all doc tokens Dj)
  Final score = sum(score_i for all query tokens)

This captures token-level matching (query terms aligning with specific document tokens)
while keeping encoding independent (can pre-compute document embeddings).
```

**Why it works**:
- **Token-level matching**: Unlike two-tower (single vector per document), ColBERT captures fine-grained alignment (e.g., query term "HIPAA" aligns with specific token in document)
- **Independent encoding**: Query and document encoded separately (can pre-index documents)
- **Late interaction**: Similarity computed at query time via MaxSim (faster than cross-encoder's joint encoding)

**Comparison table**:

| Architecture | Candidates | Latency | MRR@10 | Storage | Use case |
|--------------|------------|---------|---------|---------|----------|
| **Bi-encoder (two-tower)** | 1M | 1-5ms | 35-37% | 1× (768-dim/doc) | Stage 1: Initial retrieval |
| **ColBERT (late interaction)** | 100-1000 | 10-50ms | 40.8% | 6-10× (128-dim/token) | Stage 2: Rerank 100-1000 candidates |
| **Cross-encoder** | 10-50 | 50-500ms | 42-44% | 1× (model params) | Stage 3: Final reranking |

**Accuracy**: ColBERT achieves 95-98% of cross-encoder accuracy (MRR@10: 40.8% vs 42-44%) with 10× lower latency.

**RAGatouille implementation** (simplified ColBERT API):

```python
from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Stage 1: Two-tower retrieval (fast, broad recall)
dense_model = SentenceTransformer('all-mpnet-base-v2')

# Stage 2: ColBERT reranking (medium speed, high accuracy)
colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Stage 3: Cross-encoder final reranking (slow, highest accuracy)
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def three_stage_retrieval(query: str, documents: list[str], stage1_k: int = 100, stage2_k: int = 50, final_k: int = 5):
    """
    Three-stage retrieval pipeline: Two-tower → ColBERT → Cross-encoder

    Args:
        query: User query
        documents: Full document corpus (millions of docs)
        stage1_k: Candidates after stage 1 (two-tower)
        stage2_k: Candidates after stage 2 (ColBERT)
        final_k: Final results after stage 3 (cross-encoder)

    Returns:
        Top-k documents after three-stage ranking
    """
    # Stage 1: Two-tower retrieval (1M → 100 candidates in 30-50ms)
    query_embedding = dense_model.encode(query)
    doc_embeddings = np.array([dense_model.encode(doc) for doc in documents])
    stage1_scores = np.dot(doc_embeddings, query_embedding)
    stage1_indices = np.argsort(stage1_scores)[-stage1_k:][::-1]
    stage1_docs = [documents[i] for i in stage1_indices]

    # Stage 2: ColBERT reranking (100 → 50 candidates in 50-100ms)
    # RAGatouille's rerank() uses MaxSim internally
    colbert_results = colbert_model.rerank(query=query, documents=stage1_docs, k=stage2_k)
    stage2_docs = [result["content"] for result in colbert_results]

    # Stage 3: Cross-encoder final reranking (50 → 5 in 50-100ms)
    pairs = [[query, doc] for doc in stage2_docs]
    stage3_scores = cross_encoder.predict(pairs)
    stage3_indices = np.argsort(stage3_scores)[-final_k:][::-1]

    return [stage2_docs[i] for i in stage3_indices]


# Usage
documents = load_documents()  # 1M documents

# Three-stage pipeline: 1M → 100 → 50 → 5
final_results = three_stage_retrieval(
    query="What are HIPAA's privacy requirements?",
    documents=documents,
    stage1_k=100,   # Two-tower: 1M → 100 (30-50ms)
    stage2_k=50,    # ColBERT: 100 → 50 (50-100ms)
    final_k=5       # Cross-encoder: 50 → 5 (50-100ms)
)

# Total latency: 130-250ms (within 300ms SLA for interactive RAG)
```

**When to use ColBERT**:
- **Reranking 100-1000 candidates**: Cross-encoder too slow (50-500ms), two-tower not accurate enough
- **Latency budget 50-200ms** for stage 2: Interactive RAG systems with 300ms total SLA (stage 1: 50ms, stage 2: 100ms, stage 3: 100ms, generation: 50ms)
- **Token-level matching matters**: Multi-term queries ("HIPAA privacy rules"), acronyms (HIPAA, GDPR), named entities (proper nouns) where exact token alignment improves relevance
- **Storage available**: 6-10× larger embeddings acceptable (pre-index documents offline, store token embeddings)

**When to skip ColBERT**:
- **<100 candidates**: Cross-encoder fast enough (50ms for 50 candidates) — don't add complexity
- **>1000 candidates**: ColBERT latency grows linearly with candidates (1000 candidates = 100-500ms, too slow) — use two-tower or approximate reranking
- **Storage constrained**: 6-10× storage overhead too expensive (cloud storage costs, memory limits)
- **Latency critical**: <50ms total budget — use two-tower only or GPU-accelerated cross-encoder

**Trade-offs**:

| Dimension | Two-tower | ColBERT | Cross-encoder |
|-----------|-----------|---------|---------------|
| **Accuracy (MRR@10)** | 35-37% | 40.8% | 42-44% |
| **Latency (100 candidates)** | 1-5ms | 10-50ms | 50-500ms |
| **Storage** | 768-dim/doc (1×) | 128-dim/token (6-10×) | Model params only (1×) |
| **Pre-computation** | Yes (encode docs offline) | Yes (encode docs offline) | No (must encode query+doc jointly) |
| **Scalability** | 1M+ candidates | 100-1000 candidates | 10-100 candidates |

**Storage overhead**: ColBERT stores 128-dim embeddings per token (vs 768-dim per document for two-tower). For a 200-token document:
- Two-tower: 1 × 768-dim = 768 floats
- ColBERT: 200 × 128-dim = 25,600 floats (33× larger per document, but 6-10× after compression)

**Compression techniques** (reduce storage 2-3× with <1% accuracy loss):
- **PLAID (Pattern-based Late Interaction with Augmented Doc-level Index)**: Cluster token embeddings, store centroid IDs + residuals (2-3× smaller)
- **WARP (Weighted Average Representations of Passages)**: Learn weighted aggregation of token embeddings (2× smaller, 0.5% MRR drop)

**Production pattern** (latency-optimized three-stage):

```python
# Latency budget: 300ms total for interactive RAG
# - Stage 1 (two-tower): 30-50ms (1M → 100)
# - Stage 2 (ColBERT): 50-100ms (100 → 50)
# - Stage 3 (cross-encoder): 50-100ms (50 → 5)
# - LLM generation: 50-100ms (5 chunks → answer)
# Total: 180-350ms ✓ within 300ms SLA

def production_retrieval(query: str, stage1_k: int = 100, stage2_k: int = 50, final_k: int = 5):
    # Stage 1: Two-tower (fast, broad)
    stage1_results = dense_retrieval(query, k=stage1_k)  # 30-50ms

    # Stage 2: ColBERT (medium, high accuracy)
    stage2_results = colbert_model.rerank(query, stage1_results, k=stage2_k)  # 50-100ms

    # Stage 3: Cross-encoder (slow, highest accuracy) — optional if latency critical
    if latency_budget > 200:
        stage3_results = cross_encoder.rerank(query, stage2_results, k=final_k)  # 50-100ms
        return stage3_results
    else:
        return stage2_results[:final_k]  # Skip stage 3 if latency constrained
```

**Key insight**: ColBERT bridges the accuracy-latency gap. Use it when two-tower retrieval isn't accurate enough but cross-encoder reranking is too slow.

---

## Search-time index tuning

Vector indexes are built during indexing (see [RAG Indexing Vector Storage]({{ site.baseurl }}/docs/genai/rag/indexing/page/#vector-storage) for index types and build parameters). At search time, you tune parameters to balance recall and latency.

**HNSW efSearch tuning**: The key retrieval-time parameter for HNSW indexes.

```python
import faiss

# Load index built during indexing phase
index = faiss.read_index("index.faiss")

# Tune search-time recall/latency trade-off
index.hnsw.efSearch = 50  # Default 16, increase for better recall

# efSearch values:
# - efSearch=16: 90% recall, <5ms (speed priority)
# - efSearch=50: 95% recall, <10ms (balanced, default for RAG)
# - efSearch=100: 98% recall, <20ms (recall priority)
```

**When to optimize for recall**: Set efSearch=100. Missing the correct chunk in stage 1 means it won't be reranked in stage 2. High stage 1 recall is critical for two-stage pipelines.

**When to optimize for latency**: Set efSearch=20. Accept 90-92% recall, rely on stage 2 reranking to fix approximate errors. Only viable if you're reranking 100+ candidates.

---

## Query optimization

Queries are not always well-formed. Users ask "it", "more on that", or complex multi-part questions. Query optimization rewrites queries before retrieval.

### Query expansion

Add synonyms or related terms to increase recall.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Use T5 to generate query expansions
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def expand_query(query: str) -> str:
    """Generate 3 paraphrases of the query, concatenate for expanded retrieval."""
    input_text = f"paraphrase: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_length=50, num_return_sequences=3, num_beams=5)
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Concatenate original + paraphrases
    expanded = query + " " + " ".join(paraphrases)
    return expanded

# Original query
query = "What is RAG?"

# Expanded query for retrieval
expanded_query = expand_query(query)
# "What is RAG? What does RAG mean? Explain RAG. Define RAG."

# Retrieve with expanded query (higher recall, some noise)
results = hybrid_search(expanded_query, documents, k=50)
```

**When expansion helps**: Short queries ("RAG", "transformers") where adding context improves recall. Queries with domain-specific abbreviations ("ML" → "machine learning", "DL" → "deep learning").

**When expansion hurts**: Long, specific queries where expansions add noise. Over-expansion can dilute the original intent.

### Conversational query rewriting

RAG systems are increasingly conversational (chatbots, assistants). Users ask follow-up questions that depend on previous turns. The raw query is incomplete without context. **Conversational query rewriting** reformulates queries into standalone questions by incorporating chat history.

#### The problem

**User conversation**:
```
User: "What is RAG?"
Assistant: "RAG stands for Retrieval-Augmented Generation..."

User: "What are its limitations?"
```

**What happens without rewriting**:
- Raw query: "What are its limitations?"
- Vector search fails: "its" has no semantic meaning, embeddings don't know what "its" refers to
- Retrieval returns irrelevant chunks about limitations of unrelated topics (databases, APIs, etc.)
- LLM generates generic or wrong answer

**What happens with rewriting**:
- Rewritten query: "What are the limitations of RAG (Retrieval-Augmented Generation)?"
- Vector search succeeds: Full context embedded, retrieves chunks about RAG limitations
- LLM generates correct answer

#### How it works

Use an LLM to rewrite the query by incorporating recent chat history (last 2-5 turns).

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

def rewrite_conversational_query(query: str, chat_history: list[dict]) -> str:
    """
    Rewrite query to be standalone by incorporating chat history.

    Args:
        query: Current user query (may reference previous context)
        chat_history: List of {"role": "user"|"assistant", "content": str}

    Returns:
        Standalone query that can be understood without chat history
    """
    # Format chat history (last 5 turns)
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    history_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in recent_history
    ])

    prompt = PromptTemplate.from_template("""
Given the conversation history below, rewrite the user's latest query into a standalone question that can be understood without the conversation context.

Conversation history:
{history}

Latest user query: {query}

Rewritten standalone query:
""")

    response = llm.invoke(prompt.format(history=history_text, query=query))
    rewritten_query = response.content.strip()

    return rewritten_query

# Example usage
chat_history = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation, a technique that enhances LLMs by retrieving relevant documents before generating answers."},
    {"role": "user", "content": "What are its limitations?"},
]

query = "What are its limitations?"
rewritten = rewrite_conversational_query(query, chat_history)
# Output: "What are the limitations of RAG (Retrieval-Augmented Generation)?"

# Now use rewritten query for retrieval
results = hybrid_search(rewritten, documents, k=50)
```

#### Production patterns

**Caching rewritten queries**: Rewriting adds 100-300ms latency (LLM call). Cache rewritten queries by (query + chat_history_hash) to avoid re-rewriting identical conversations.

```python
import hashlib
import json

def get_chat_history_hash(chat_history: list[dict]) -> str:
    """Generate stable hash of chat history for caching."""
    history_str = json.dumps(chat_history, sort_keys=True)
    return hashlib.md5(history_str.encode()).hexdigest()

def rewrite_with_cache(query: str, chat_history: list[dict], cache: dict) -> str:
    """Rewrite query with caching to reduce latency."""
    cache_key = f"{query}:{get_chat_history_hash(chat_history)}"

    if cache_key in cache:
        return cache[cache_key]

    rewritten = rewrite_conversational_query(query, chat_history)
    cache[cache_key] = rewritten
    return rewritten
```

**Detecting when rewriting is needed**: Not all queries need rewriting. Standalone queries ("What is RAG?", "Explain HIPAA compliance") don't reference previous context. Skip rewriting to save latency.

```python
def needs_rewriting(query: str) -> bool:
    """
    Heuristic: Does query contain pronouns or references that need context?

    Returns True if query likely needs chat history context.
    """
    # Pronouns and references that signal conversational dependency
    contextual_patterns = [
        r'\b(it|its|this|that|these|those|they|them|their)\b',  # Pronouns
        r'\b(also|too|as well|additionally|furthermore)\b',     # Additive references
        r'\b(instead|however|but|although)\b',                  # Contrastive references
        r'\b(the same|similar|different|compared)\b',           # Comparative references
    ]

    import re
    for pattern in contextual_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    return False

# Usage
query1 = "What are its limitations?"  # needs_rewriting() → True
query2 = "What is RAG?"               # needs_rewriting() → False

if needs_rewriting(query1):
    rewritten = rewrite_conversational_query(query1, chat_history)
    results = hybrid_search(rewritten, documents, k=50)
else:
    results = hybrid_search(query2, documents, k=50)  # Use original query
```

#### When conversational rewriting matters

**Critical for**:
- Chatbot interfaces (every query after the first may reference context)
- Multi-turn Q&A (user drilling down: "What is X?" → "How does it work?" → "What are the alternatives?")
- Voice assistants (users speak naturally: "Tell me about RAG. What are its benefits? How do I implement it?")

**Less important for**:
- Single-turn search (Google-style: each query is independent)
- Keyword search (users already provide full context: "RAG limitations", "HIPAA compliance requirements")
- FAQ systems (questions are standalone: "How do I reset my password?")

#### Latency considerations

**Latency breakdown**:
- Rewriting (LLM call): 100-300ms
- Retrieval (after rewriting): 60-130ms
- **Total**: 160-430ms (vs 60-130ms without rewriting)

**Optimization strategies**:
- Use smaller/faster model for rewriting (GPT-3.5-turbo, Claude Haiku): 50-100ms (vs 100-300ms for GPT-4)
- Cache rewritten queries (40-60% hit rate for FAQ/support): Reduce effective latency by 40-60%
- Async rewriting: Start rewriting and retrieval in parallel if you can predict when rewriting is needed
- Skip rewriting when not needed (use heuristic above): Save 100-300ms on 50-70% of queries

#### Advanced: Fine-tuned rewriting model

For high-volume production systems, fine-tune a small model for query rewriting instead of using GPT-4.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Fine-tune T5-small on (chat_history, query, rewritten_query) triplets
# Training data: 1000-5000 examples of conversational queries + rewrites

model = T5ForConditionalGeneration.from_pretrained("your-org/t5-query-rewriter")
tokenizer = T5Tokenizer.from_pretrained("your-org/t5-query-rewriter")

def rewrite_fast(query: str, chat_history: list[dict]) -> str:
    """
    Fast query rewriting with fine-tuned T5-small (10-20ms vs 100-300ms for GPT-4).
    """
    # Format input
    history_text = " ".join([msg['content'] for msg in chat_history[-3:]])
    input_text = f"rewrite query with context: {history_text} [SEP] {query}"

    # Generate rewrite
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return rewritten

# Latency: 10-20ms (vs 100-300ms for GPT-4)
# Quality: 90-95% of GPT-4 quality (acceptable for most use cases)
```

**Trade-off**: Fine-tuning costs upfront (1-2 weeks engineering + labeled data), but saves 100-200ms per query at scale (1M queries/day = $10-50K/year in API costs).

### Query decomposition

Break multi-part questions into sub-queries, retrieve separately, merge results.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

def decompose_query(query: str) -> list[str]:
    """Decompose complex query into simple sub-queries."""
    prompt = PromptTemplate.from_template("""
Break this question into 2-4 simple sub-questions that together answer the original question.

Question: {query}

Sub-questions (one per line):
""")

    response = llm.invoke(prompt.format(query=query))
    sub_queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    return sub_queries

# Complex query
query = "Compare the privacy requirements of HIPAA and GDPR and explain how they differ for healthcare data."

# Decompose
sub_queries = decompose_query(query)
# 1. What are HIPAA's privacy requirements?
# 2. What are GDPR's privacy requirements?
# 3. How do HIPAA and GDPR differ for healthcare data?

# Retrieve for each sub-query
all_results = []
for sub_q in sub_queries:
    results = hybrid_search(sub_q, documents, k=20)
    all_results.extend(results)

# Deduplicate and rerank
unique_docs = list(set(all_results))
top_chunks = rerank(query, unique_docs, k=5)  # Rerank with original query
```

**When decomposition helps**: Multi-hop questions where a single retrieval misses parts. "Compare X and Y" often needs separate retrievals for X and Y.

**When decomposition hurts**: Simple queries where decomposition adds latency (LLM call + multiple retrievals). Overhead is 200-500ms.

### Metadata filtering and boosting

Metadata enables both filtering (restrict search space) and boosting (adjust relevance scores).

#### Filtering

Filter by document metadata before retrieval to reduce search space and improve precision.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", port=6333)

# Query with metadata filter: only search recent documents
query_embedding = model.encode(query)

results = client.search(
    collection_name="documents",
    query_vector=query_embedding.tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="healthcare")),
            FieldCondition(key="created_at", range={"gte": "2024-01-01"}),
        ]
    ),
    limit=50,
)

# Only retrieves chunks from healthcare docs created in 2024+
```

**When filtering helps**: Users specify constraints ("recent research", "Python examples", "legal documents"). Time-sensitive queries where old information is wrong. Domain-specific search where categories reduce noise.

**When filtering hurts**: Filters are too restrictive (no results). Metadata is noisy or incomplete (many documents untagged).

#### Hard security filtering (RBAC)

**Distinguish hard filtering from soft boosting**: The score boosting section below uses `access_level` metadata to adjust rankings (soft boosting) — unauthorized users still see restricted content with lower scores. Hard security filtering physically blocks unauthorized chunks from retrieval results, ensuring they never reach the LLM or user. This is critical for enterprise RAG systems handling sensitive data (HIPAA, GDPR, multi-tenant SaaS).

**Why RBAC matters: Business value and risk**

Understanding RBAC requires more than technical knowledge — it's fundamentally a business risk management decision. For decision-makers evaluating whether to implement RBAC in their RAG system, consider these factors:

**Cost of data breaches without RBAC**:
- **Average breach cost**: $4.45 million (IBM Cost of a Data Breach Report 2023)
- **Customer churn**: 60% of users stop using service after a breach
- **Remediation time**: 3-6 months to restore security, rebuild customer trust, and address regulatory violations
- **Reputation damage**: Long-term brand erosion, executive turnover, PR crisis management costs

**Regulatory compliance penalties**:
- **HIPAA violations**: $100 to $1.5 million per incident (healthcare data breaches)
- **GDPR violations**: Up to €20 million or 4% of global annual revenue, whichever is higher
- **SOX violations**: SEC enforcement actions, potential criminal charges for executives
- **PCI-DSS violations**: Fines up to $100,000 per month, loss of payment processing privileges

**Business risks without RBAC**:
- **Multi-tenant data leakage**: Customer A retrieves Customer B's sensitive data in RAG results, leading to immediate contract breach and customer churn
- **Regulatory violations**: Failed HIPAA/GDPR audit results in fines, business suspension, and loss of enterprise customers
- **Insider threats**: Employees access data beyond their role (HR accessing executive compensation, junior analyst accessing M&A deal flow)
- **Competitive intelligence leaks**: Employees accidentally or intentionally retrieve competitor client files, trade secrets, or strategic documents

**Market access without RBAC**:
- **Lost opportunities**: Cannot serve regulated industries (healthcare, finance, legal, government) that require vendor security compliance
- **Contract requirements**: Enterprise customers require SOC 2 Type II, ISO 27001, or security questionnaires demonstrating access controls
- **Competitive disadvantage**: Competitors with RBAC win enterprise deals you cannot bid on

**ROI calculation for implementing RBAC**:

| Cost/Value Item | Amount |
|----------------|--------|
| **Implementation cost** | $50,000 - $150,000 (identity integration, metadata audit, compliance monitoring) |
| **Annual operational cost** | $10,000 - $30,000 (monitoring, compliance audits, maintenance) |
| **Value protected: Breach prevention** | $2,000,000 - $5,000,000+ (average breach cost avoided) |
| **Value protected: Compliance** | $100,000 - $20,000,000 (HIPAA/GDPR penalties avoided) |
| **Value protected: Market access** | Variable (healthcare, finance, legal customers require RBAC for contracts) |
| **Payback period** | < 6 months for regulated industries |
| **Net ROI** | 400% - 5,000% in first year |

**When the business case is strongest**:
- Multi-tenant SaaS platforms with enterprise customers
- Regulated industries requiring demonstrable access controls
- Organizations handling PII, PHI, financial records, or trade secrets
- Companies pursuing enterprise contracts (SOC 2, ISO 27001 required)
- High insider threat environments (large organizations, sensitive data)

**Architecture**: Hard filtering implements Role-Based Access Control (RBAC) at query time:

1. **User authentication**: System extracts user identity from JWT token or session
2. **Permission resolution**: Resolve user's roles, departments, and permissions
3. **Query-time filtering**: Vector database filters chunks where user's roles intersect with chunk's `roles` metadata
4. **Allowlist-based**: Only chunks matching user's permissions are retrieved (default deny)

**Qdrant implementation** (most vector databases support similar filtering):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
import jwt

def secure_retrieval(query: str, user_token: str, k: int = 10):
    """
    RBAC-filtered retrieval: Only return chunks user is authorized to see.

    Args:
        query: User query text
        user_token: JWT token containing user identity and roles
        k: Number of results to return

    Returns:
        List of authorized chunks (unauthorized chunks physically blocked)
    """
    # 1. Validate token and extract user permissions
    try:
        payload = jwt.decode(user_token, SECRET_KEY, algorithms=["HS256"])
        user_roles = payload.get("roles", [])  # e.g., ["engineering", "compliance"]
        user_dept = payload.get("department", "")  # e.g., "legal"
        user_id = payload.get("user_id", "")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid authentication token")

    # 2. Embed query
    query_embedding = model.encode(query).tolist()

    # 3. Build RBAC filter (allowlist: chunk roles ∩ user roles ≠ ∅)
    rbac_filter = Filter(
        must=[
            # User must have at least one role that matches chunk's authorized roles
            FieldCondition(
                key="roles",
                match=MatchAny(any=user_roles)  # Chunk accessible if ANY user role in chunk's roles list
            )
        ]
    )

    # Optional: Add hierarchical department filtering
    # if user_dept:
    #     rbac_filter.must.append(
    #         FieldCondition(key="departments", match=MatchAny(any=[user_dept]))
    #     )

    # 4. Query with hard security filter
    client = QdrantClient(host="localhost", port=6333)
    results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        query_filter=rbac_filter,  # CRITICAL: Filter applied before similarity scoring
        limit=k,
    )

    return [
        {
            "text": hit.payload["text"],
            "score": hit.score,
            "source": hit.payload.get("source", ""),
            "roles": hit.payload.get("roles", []),  # Transparency: Show why user can see this
        }
        for hit in results
    ]

# Example usage
user_token = generate_jwt(user_id="user_123", roles=["engineering", "compliance"], department="legal")
results = secure_retrieval(
    query="What are our HIPAA compliance requirements?",
    user_token=user_token,
    k=10
)

# User with roles ["engineering", "compliance"] ONLY sees chunks with:
# - roles: ["engineering", ...]  ✓
# - roles: ["compliance", ...]  ✓
# - roles: ["legal", "hr"]  ✗ (physically blocked, never retrieved)
```

**Hierarchical permissions**: Structure RBAC metadata hierarchically for flexible filtering:

```python
# Department-level (broad access)
metadata = {
    "roles": ["legal"],
    "departments": ["legal"],
}

# Team-level (medium access)
metadata = {
    "roles": ["legal", "compliance"],
    "departments": ["legal"],
    "team": "privacy-team",
}

# Individual-level (narrow access)
metadata = {
    "roles": ["legal"],
    "departments": ["legal"],
    "owner_id": "user_456",  # Only owner can access
}

# Query filters can target any level:
# - Broad: FieldCondition(key="departments", match=MatchAny(any=["legal"]))
# - Narrow: FieldCondition(key="owner_id", match=MatchValue(value=user_id))
```

**When to use hard filtering: Concrete business scenarios**

Understanding when RBAC is required goes beyond abstract security principles — here are concrete scenarios business stakeholders will recognize:

**Healthcare: HIPAA compliance**
- **Scenario**: A cardiologist can retrieve cardiology patient records in the RAG system but must be physically blocked from accessing psychiatry records, even within the same hospital system. HIPAA requires role-based access controls to protect patient privacy across departments.
- **Business impact**: HIPAA violation = $100 to $1.5 million penalty per incident + potential criminal charges for executives + loss of Medicare/Medicaid reimbursement eligibility
- **Implementation**: Role-based filtering where `cardiologist` role → only chunks with `department: cardiology` metadata. Psychiatry records are physically blocked from retrieval results.

**Multi-tenant SaaS: Customer data isolation**
- **Scenario**: Customer A (Acme Corp) queries the RAG system for "Q3 sales projections" and must never see Customer B's (Beta Inc) proprietary sales data, even if the query semantics are similar. Data leakage between tenants = contract breach + immediate customer churn.
- **Business impact**: Data leakage between tenants = 100% customer churn + lawsuit for breach of contract + reputation damage preventing future enterprise sales
- **Implementation**: `customer_id` metadata filtering where user's JWT token contains `customer_id: acme_corp` → only chunks with matching `customer_id` are retrieved. Beta Inc chunks are physically blocked.

**Legal: Attorney-client privilege and conflict walls**
- **Scenario**: Attorney handling Case A (tech startup acquisition) cannot access Case B (competitor's lawsuit) discovery documents, even though both are stored in the firm's document vault. Bar association ethics rules require "Chinese walls" between conflicting representations.
- **Business impact**: Conflict of interest = malpractice lawsuit + state bar disbarment proceedings + loss of client trust + inability to represent either party
- **Implementation**: `case_id` + `role` filtering where attorney's permissions include `case_id: ["case_a"]` → only Case A chunks retrieved. Case B chunks physically blocked even if semantically relevant.

**HR: Employee privacy and data sovereignty**
- **Scenario**: US HR administrator cannot access EU employee compensation data (GDPR Article 48: cross-border data transfers require legal basis). HR manager cannot see CEO compensation. Employee cannot see peer performance reviews.
- **Business impact**: GDPR violation = up to 4% of global annual revenue (€20 million cap) + individual lawsuits from affected employees + regulatory investigation
- **Implementation**: Hierarchical permissions with `region` + `role` + `data_classification` metadata. US HR admin role → blocked from `region: EU` chunks. Non-executive HR → blocked from `data_classification: executive_compensation`.

**Financial services: Regulatory compliance and insider trading prevention**
- **Scenario**: Junior analyst can retrieve market data and public SEC filings but must be blocked from accessing internal deal flow, M&A target lists, or executive compensation. SOC 2 Type II audit requires proof of access controls and audit logs.
- **Business impact**: Failed audit = loss of enterprise customers requiring SOC 2 compliance + SEC investigation for insider trading if privileged information leaks + competitive intelligence exposure
- **Implementation**: Role-based + data classification filtering where `analyst` role → only `data_classification: public` or `data_classification: market_data` chunks. `data_classification: internal_deal_flow` physically blocked.

**When you MUST implement RBAC**:
- **Multi-tenant SaaS**: Different customers share the same RAG infrastructure
- **Regulated industries**: Healthcare (HIPAA), finance (SOX, SEC), EU users (GDPR), payments (PCI-DSS)
- **Sensitive data types**: PII, PHI, financial records, legal documents, HR data, trade secrets, customer lists
- **Enterprise contracts**: Customers require SOC 2 Type II, ISO 27001, or security questionnaire attestations
- **Insider threat risk**: Large organizations where employees should not access data beyond their role

**When soft boosting is sufficient** (but understand the risks):
- **Single-tenant systems**: One organization, all users trusted with all data (no customer isolation needed)
- **Public data only**: No sensitive, regulated, or confidential information in the RAG system
- **Non-enterprise customers**: Consumer app with no compliance requirements or security questionnaires
- **Controlled environment**: Internal tool with <10 highly trusted users in a small organization

**Cost of skipping RBAC when you need it**:
- **Data breach**: $4.45 million average cost + 3-6 months remediation + 60% customer churn
- **Compliance violation**: $100,000 to $20 million in regulatory penalties (HIPAA, GDPR, SOX)
- **Customer churn**: Immediate termination of enterprise contracts after data leakage incident
- **Lost market access**: Cannot bid on healthcare, finance, legal, or government contracts requiring security compliance
- **Reputation damage**: Long-term trust erosion, inability to sign new enterprise customers, executive turnover

**Trade-offs: Technical and business considerations**

**Technical trade-offs**:
- **Latency**: +5-15ms per query for metadata filtering (vector databases index metadata for fast lookups)
- **Recall**: Filtered chunks are unavailable by design (security > recall). Users may miss relevant content they're not authorized for.
- **Implementation complexity**: Requires JWT token management, permission synchronization between identity provider and vector database, and careful metadata design at indexing time.
- **Index size**: Adding RBAC metadata (roles, departments, owner_id) increases storage by 5-10% (minimal compared to embeddings).

**Business trade-offs: Cost-benefit analysis**

| Dimension | Without RBAC | With RBAC |
|-----------|--------------|-----------|
| **Implementation cost** | $0 upfront | $50,000 - $150,000 (IdP integration, metadata audit, monitoring setup) |
| **Operational cost** | $0/month | +5-10% storage, +5-15ms latency per query, $10K-30K/year monitoring |
| **Breach risk** | High (no access controls) | Low (physical blocking prevents unauthorized access) |
| **Breach cost** | $4.45M average + customer churn | Prevented (or scope limited to single user's authorized data) |
| **Compliance penalties** | $100K-$1.5M per HIPAA violation<br>Up to €20M or 4% revenue GDPR | Compliant (penalty risk eliminated) |
| **Market access** | Cannot serve regulated industries | Can serve healthcare, finance, legal (enterprise contracts unlocked) |
| **Customer trust** | Vulnerable (no security differentiation) | Strong (enterprise-grade security = competitive advantage) |
| **Time to implement** | N/A | 6-12 weeks (depends on IdP complexity and metadata audit scope) |
| **Payback period** | N/A | <6 months for regulated industries, 12-18 months for unregulated |

**Business ROI calculation**:
- **Upfront investment**: $50,000 - $150,000 (one-time: identity provider integration, metadata audit, compliance monitoring setup)
- **Annual operational cost**: $10,000 - $30,000 (ongoing: monitoring tools, compliance audits, permission sync maintenance)
- **Value protected**: $2M - $5M+ in breach prevention + $100K - $20M in regulatory compliance + market access to regulated industries
- **Net ROI**: 400% - 5,000% in first year (for regulated industries or multi-tenant SaaS)

**When ROI justifies implementation**:
- **High**: Multi-tenant SaaS, healthcare, finance, legal, government (immediate compliance requirement + high breach cost)
- **Medium**: Large enterprises with sensitive internal data (insider threat prevention + audit requirements)
- **Low**: Single-tenant consumer apps with public data (no compliance requirement, low breach risk)

**Alternatives**: Pinecone, Weaviate, Milvus support similar metadata filtering. Qdrant's `Filter` API is representative — adapt syntax to your vector database's query language.

**Pitfalls**:
- **Missing RBAC metadata**: If you forget to add `roles` at indexing time, you can't filter at query time without reindexing.
- **Token validation**: Always validate JWT signatures and expiration. Trusting client-provided tokens without verification is a critical vulnerability.
- **Permission drift**: If user permissions change in your identity provider (Okta, Auth0) but not in vector database metadata, users may access stale content. Sync permissions regularly or use short-lived tokens (1-hour TTL).

**Audit trails, monitoring, and compliance reporting**

Implementing RBAC is only the first step — regulatory compliance and security operations require proof that access controls are enforced and monitored. Here's how to build audit-ready RBAC systems:

**Why audit trails matter**:
- **Regulatory requirement**: HIPAA, SOC 2, GDPR, PCI-DSS, ISO 27001 require evidence of access controls and audit logs
- **Security monitoring**: Detect unauthorized access attempts, insider threats, permission drift, and anomalous query patterns
- **Incident response**: Answer "Who accessed the leaked document before the breach?" during forensic investigation
- **Compliance reporting**: Generate audit evidence for regulators, external auditors, and enterprise customer security questionnaires

**What to log in every retrieval request**:

```python
from datetime import datetime
import logging

def secure_retrieval_with_audit(query: str, user_token: str, k: int = 10):
    """
    RBAC-filtered retrieval with comprehensive audit logging.
    """
    # 1. Validate token and extract user permissions
    try:
        payload = jwt.decode(user_token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id", "")
        user_roles = payload.get("roles", [])
        user_dept = payload.get("department", "")
    except jwt.InvalidTokenError:
        # Log failed authentication attempt
        audit_log_failed_auth(user_token, query)
        raise ValueError("Invalid authentication token")

    # 2. Execute RBAC-filtered retrieval (same as before)
    query_embedding = model.encode(query).tolist()
    rbac_filter = Filter(
        must=[
            FieldCondition(key="roles", match=MatchAny(any=user_roles))
        ]
    )

    client = QdrantClient(host="localhost", port=6333)
    results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        query_filter=rbac_filter,
        limit=k,
    )

    # 3. Count total chunks (without filter) to measure blocking effectiveness
    total_results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=k,
    )
    chunks_blocked = len(total_results) - len(results)

    # 4. Log audit trail (send to SIEM for centralized monitoring)
    audit_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "rag_retrieval",
        "user_id": user_id,
        "user_roles": user_roles,
        "user_department": user_dept,
        "query": query,  # May need to sanitize for PII (redact SSNs, emails)
        "chunks_returned": len(results),
        "chunks_blocked": chunks_blocked,
        "blocking_rate": chunks_blocked / len(total_results) if total_results else 0,
        "ip_address": request.remote_addr,  # Requires Flask/Django request context
        "session_id": request.session.get("session_id", ""),
        "user_agent": request.headers.get("User-Agent", ""),
    }

    # Send to SIEM (Splunk, ELK, Datadog) for centralized monitoring
    log_to_siem(audit_log)

    # Also log locally for debugging
    logging.info(f"RAG retrieval: user={user_id}, chunks_returned={len(results)}, chunks_blocked={chunks_blocked}")

    return [
        {
            "text": hit.payload["text"],
            "score": hit.score,
            "source": hit.payload.get("source", ""),
            "roles": hit.payload.get("roles", []),
        }
        for hit in results
    ]
```

**Security monitoring and alerting**:

Configure alerts in your SIEM to detect suspicious access patterns:

- **Alert: Unusual access volume**: "User accessed 10× more chunks than typical daily average" (potential data exfiltration)
- **Alert: Repeated failed access**: "User queried restricted chunks 5+ times in 1 hour with 100% blocking rate" (unauthorized access attempt)
- **Alert: Permission drift detected**: "User's roles changed in IdP (Okta) but vector DB metadata not updated in 24 hours" (stale permissions)
- **Alert: Sensitive query from unauthorized role**: "Non-legal user queried 'confidential legal memo' with 0 results" (reconnaissance for privilege escalation)
- **Alert: High blocking rate**: "User's queries blocked 80%+ of relevant chunks consistently" (user needs additional permissions or is querying out-of-scope data)

**Compliance reporting for audits**:

Different regulatory frameworks require specific evidence:

**SOC 2 Type II (Trust Services Criteria CC6.1: Logical Access)**:
- **Requirement**: Demonstrate access controls are enforced consistently over 12-month audit period
- **Evidence**: Generate report showing RBAC enforcement metrics (queries processed, chunks blocked, unauthorized access attempts) with monthly aggregation
- **Audit query**: "Show me audit logs proving user A with role 'analyst' was blocked from accessing 'executive_compensation' data classification"

**HIPAA Security Rule § 164.308(a)(4) (Access Control)**:
- **Requirement**: Implement access controls to limit PHI access to authorized users, plus audit controls to record and examine access
- **Evidence**: Audit logs showing user ID, timestamp, PHI accessed (document ID), and authorization basis (role)
- **Audit query**: "Prove that Dr. Smith (cardiologist) cannot access psychiatry patient records via query logs showing 0 psychiatry chunks returned"

**GDPR Article 30 (Records of Processing Activities)**:
- **Requirement**: Maintain records of data processing activities including access controls and security measures
- **Evidence**: Document RBAC policy (roles, permissions), plus audit logs showing enforcement over time
- **Audit query**: "Show data processing records for EU employee data, proving US employees cannot access it"

**ISO 27001 (A.9.1 Access Control)**:
- **Requirement**: Document access control policy and demonstrate technical enforcement
- **Evidence**: RBAC policy document + technical implementation (code review) + audit logs proving enforcement
- **Audit query**: "Demonstrate that access control policy documented in ISMS matches technical implementation in RAG system"

**Testing RBAC enforcement**:

Regular testing ensures RBAC works as designed:

- **Penetration testing**: "Can user A manipulate JWT token to access user B's data?" (test token validation, signature verification)
- **Regression testing**: "After reindexing documents, verify RBAC filters still work for all user roles" (test metadata preservation)
- **Quarterly access audits**: "Sample 100 random queries, verify correct filtering for each user's roles" (statistical validation)
- **Red team exercise**: "Attempt to exfiltrate sensitive data using RAG system" (test defense-in-depth)

**Retention and storage**:
- **Audit logs**: Retain for 1-7 years (depends on regulatory requirements: HIPAA 6 years, SOC 2 1 year, GDPR 3 years)
- **Storage**: Use append-only audit log storage (prevents tampering) with encryption at rest
- **Access controls**: Limit audit log access to security team and compliance officers (prevent log manipulation by unauthorized users)

**Cross-references**: See [indexing metadata strategy]({{ site.baseurl }}/docs/genai/rag/indexing/#metadata-strategy) for how to structure RBAC fields at indexing time.

---

#### Score boosting

Adjust retrieval scores based on metadata to promote authoritative, recent, or contextually relevant documents.

```python
from datetime import datetime, timedelta
import numpy as np

def metadata_boosted_search(query: str, documents: list[dict], k: int = 10):
    """
    Retrieve with metadata-based score boosting.

    Boosts:
    - Recency: Recent documents ranked higher
    - Authority: Documents from trusted authors ranked higher
    - User context: Documents matching user's department/role ranked higher
    """
    # Base retrieval (semantic similarity)
    query_embedding = model.encode(query)
    doc_embeddings = np.array([model.encode(doc['text']) for doc in documents])
    base_scores = np.dot(doc_embeddings, query_embedding)

    # Normalize to [0, 1]
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + 1e-8)

    # Compute boost factors
    boosted_scores = base_scores.copy()

    for i, doc in enumerate(documents):
        # Recency boost: Exponential decay (recent = higher boost)
        doc_age_days = (datetime.utcnow() - datetime.fromisoformat(doc['created_at'])).days
        recency_boost = np.exp(-doc_age_days / 365)  # Decay over 1 year
        boosted_scores[i] *= (1 + 0.2 * recency_boost)  # Up to 20% boost for recent docs

        # Authority boost: Trusted sources get +10%
        if doc.get('author') in ['legal-team', 'compliance-team']:
            boosted_scores[i] *= 1.1

        # Category boost: Match user's domain
        if doc.get('category') == user_context['department']:  # e.g., 'engineering'
            boosted_scores[i] *= 1.15

    # Return top-k
    top_indices = np.argsort(boosted_scores)[::-1][:k]
    results = [documents[i] for i in top_indices]

    return results
```

**Recency boosting**: Use exponential decay — documents lose relevance over time. Tune decay rate per domain (news: fast decay, legal docs: slow decay).

**Authority boosting**: Promote documents from trusted sources. Encode authority as metadata (author, domain, citation count). Boost by 10-30% based on trust level.

**Contextual boosting**: Match user context (role, department, permissions). Engineering queries prioritize technical docs, legal queries prioritize compliance docs.

**When boosting helps**: Time-sensitive domains (news, regulations). Multi-domain corpora where authority varies. Personalized search where user context matters.

**When boosting hurts**: Over-boosting drowns semantic relevance. A recent irrelevant doc shouldn't beat an older highly relevant doc. Keep boosts modest (10-30%, not 2-3×).

#### Positional boosting

Document structure matters. Introduction and conclusion sections often contain key concepts, definitions, and summaries. Middle sections contain details. Boosting chunks based on position improves precision for conceptual queries.

**Lost in the middle problem**: LLMs exhibit U-shaped performance with primacy (beginning) and recency (end) bias. Performance drops >20% when relevant information is in the middle of long contexts. This affects both which chunks to retrieve and how to order them.

```python
def positional_boosted_search(query: str, documents: list[dict], k: int = 10):
    """
    Boost chunks based on document position (intro/conclusion preference).

    Boosts:
    - First chunk (introduction): +15-20%
    - Last chunk (conclusion): +10-15%
    - Section headers: +10%
    - Middle chunks: No boost (or slight penalty)
    """
    # Base retrieval
    query_embedding = model.encode(query)
    doc_embeddings = np.array([model.encode(doc['text']) for doc in documents])
    base_scores = np.dot(doc_embeddings, query_embedding)

    # Normalize to [0, 1]
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + 1e-8)

    # Apply positional boosts
    boosted_scores = base_scores.copy()

    for i, doc in enumerate(documents):
        chunk_position = doc.get('chunk_index', 0)
        total_chunks = doc.get('total_chunks', 1)

        # First chunk (introduction) - strongest boost
        if chunk_position == 0:
            boosted_scores[i] *= 1.2  # +20%

        # Last chunk (conclusion) - second strongest boost
        elif chunk_position == total_chunks - 1:
            boosted_scores[i] *= 1.15  # +15%

        # Section header chunks
        if doc.get('is_section_header', False):
            boosted_scores[i] *= 1.1  # +10%

        # Middle chunks - no boost or slight penalty for very long docs
        # This addresses "lost in the middle" by de-prioritizing middle content
        if total_chunks > 10:
            # Apply U-shaped curve: penalize middle chunks
            normalized_position = chunk_position / (total_chunks - 1)  # 0 to 1
            # U-curve: min at 0.5 (middle), max at 0 (start) and 1 (end)
            position_weight = 1 - 0.15 * (1 - 4 * (normalized_position - 0.5) ** 2)
            boosted_scores[i] *= position_weight

    # Return top-k
    top_indices = np.argsort(boosted_scores)[::-1][:k]
    results = [documents[i] for i in top_indices]

    return results
```

**Section-based weighting**: Different document types have different valuable sections. Research papers: methodology and results matter most. Technical docs: implementation sections beat introductions. Legal documents: definitions and requirements sections are critical.

```python
# Section type weights by document category
SECTION_WEIGHTS = {
    'research_paper': {
        'abstract': 1.15,
        'introduction': 1.1,
        'methodology': 1.25,  # Most important for technical understanding
        'results': 1.2,
        'discussion': 1.15,
        'conclusion': 1.1,
        'references': 0.8,  # Less relevant for most queries
    },
    'technical_doc': {
        'overview': 1.1,
        'installation': 1.0,
        'implementation': 1.3,  # Code examples are gold
        'api_reference': 1.2,
        'troubleshooting': 1.15,
        'faq': 1.1,
    },
    'legal_doc': {
        'definitions': 1.3,  # Critical for understanding
        'requirements': 1.25,
        'obligations': 1.25,
        'exceptions': 1.2,
        'penalties': 1.15,
        'preamble': 0.9,
    },
}

def apply_section_weights(doc: dict, base_score: float) -> float:
    """Apply document-type-specific section weights."""
    doc_type = doc.get('doc_type', 'unknown')
    section = doc.get('section', 'unknown').lower()

    weights = SECTION_WEIGHTS.get(doc_type, {})
    weight = weights.get(section, 1.0)

    return base_score * weight
```

**Chunk ordering (lost in the middle mitigation)**: After retrieval and reranking, order chunks to place most relevant at start and end, not middle.

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to avoid 'lost in the middle' problem.

    Strategy:
    - Place most relevant chunks at positions 0 and -1 (start and end)
    - Place medium-relevance chunks at positions 1 and -2
    - Place least relevant chunks in the middle
    """
    if len(chunks) <= 2:
        return chunks  # No reordering needed

    # Chunks already sorted by relevance (highest first)
    reordered = []

    # Alternate placing high-relevance chunks at start/end
    left_idx = 0
    right_idx = len(chunks) - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            # Place at start
            reordered.insert(left_idx, chunk)
            left_idx += 1
        else:
            # Place at end
            reordered.insert(right_idx, chunk)

    return reordered

# After retrieval and reranking
top_chunks = two_stage_retrieval(query, documents, k=5)
optimized_order = reorder_for_llm(top_chunks)
# optimized_order: [most_relevant, 3rd, 5th (least), 4th, 2nd]
# Creates U-shape: strong start, weak middle, strong end
```

**When positional boosting helps**: Documents with clear structure (papers, reports, technical docs). Conceptual queries that match introductions/conclusions ("What is X?", "How does X work?"). Domain-specific documents where section importance is known (legal, medical, academic).

**When positional boosting hurts**: Unstructured documents (chat logs, social media, transcripts). Queries seeking specific details buried in middle sections. Over-application creates bias against relevant middle content.

**Production considerations**:
- Store positional metadata during indexing (chunk_index, total_chunks, section_name, is_header)
- Tune boost values per document type (research papers need different boosts than legal docs)
- Monitor whether first/last chunk bias helps or hurts your specific use case
- Combine with reranking: apply positional boosts before cross-encoder reranking, not after
- Use chunk reordering for LLM input to mitigate "lost in the middle" effect

#### Context enrichment

Pass metadata to the LLM for attribution and context. This reduces hallucination and enables source verification.

```python
def retrieve_with_metadata(query: str, k: int = 5):
    """Retrieve chunks and format with metadata for LLM."""
    results = client.search(
        collection_name="documents",
        query_vector=model.encode(query).tolist(),
        limit=k,
    )

    # Format chunks with metadata for LLM
    context = []
    for i, result in enumerate(results):
        chunk_text = result.payload['text']
        metadata = result.payload

        # Include source, date, author in context
        context.append(f"""
[Source {i+1}]
Document: {metadata.get('source', 'unknown')}
Page: {metadata.get('page_number', 'N/A')}
Author: {metadata.get('author', 'unknown')}
Date: {metadata.get('created_at', 'unknown')}

Content:
{chunk_text}
""")

    # Send to LLM with attribution prompt
    prompt = f"""Answer the question using only the provided sources. Cite sources by number [1], [2], etc.

Sources:
{''.join(context)}

Question: {query}

Answer:"""

    return prompt

# LLM output includes: "According to [1], HIPAA requires... [2] further specifies..."
```

**Why context enrichment matters**: LLMs can cite sources, users can verify claims, you can detect when LLM extrapolates beyond retrieved context.

**Best practices**:
- Include source, page, date, author — enough for users to verify
- Format metadata clearly (Source 1, Source 2) for easy LLM citation
- Prompt LLM to cite sources explicitly ([1], [2])
- Don't overload context with too much metadata (keep < 20% of chunk text)

### Advanced query techniques

Beyond expansion and decomposition, several techniques transform the query embedding itself to improve retrieval quality.

#### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer to the query, then use that answer's embedding for retrieval instead of the raw query. Documents are written as answers, not questions — matching answer-to-answer is more effective than matching question-to-answer.

```python
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def hyde_retrieval(query: str, documents: list[str], k: int = 5):
    """
    HyDE: Generate hypothetical answer, embed it, retrieve similar documents.
    """
    # Generate hypothetical answer to the query
    hyde_prompt = f"""Given the question: "{query}"

Write a detailed, factual answer as if you had access to relevant documents. Be specific and technical."""

    hypothetical_answer = llm.invoke(hyde_prompt).content

    # Embed the hypothetical answer (not the query)
    hyde_embedding = model.encode(hypothetical_answer)

    # Retrieve documents similar to the hypothetical answer
    doc_embeddings = model.encode(documents)
    similarities = np.dot(doc_embeddings, hyde_embedding)

    top_indices = np.argsort(similarities)[::-1][:k]
    results = [documents[i] for i in top_indices]

    return results

# Query: "How does HNSW algorithm work?"
# HyDE generates: "HNSW constructs a hierarchical graph where nodes represent vectors..."
# Retrieval finds docs that match the answer structure, not the question
```

**Why this works**: Query embeddings and document embeddings live in slightly different semantic spaces. Queries are short and question-focused ("How does X work?"). Documents are long and declarative ("X works by..."). Generating a hypothetical answer bridges this gap — answer embeddings match document embeddings better than query embeddings.

**When HyDE helps**: Complex queries where the semantic gap is large. Technical questions where documents use specific terminology that queries don't. Zero-shot retrieval where the query and document vocabularies differ.

**When HyDE hurts**: The hypothetical answer is wrong (hallucinated), leading retrieval astray. Simple queries where the query already matches document language ("HIPAA compliance" matches "HIPAA compliance" directly). HyDE adds 1-3 seconds of LLM latency.

**Optimization**: Generate 3-5 hypothetical answers, average their embeddings for robustness. Use a cheaper model (GPT-3.5) for speed.

#### Multi-Query

Generate multiple query variations from different perspectives, retrieve for all, merge results with RRF or score fusion.

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def multi_query_retrieval(query: str, documents: list[str], num_queries: int = 3, k: int = 5):
    """
    Generate multiple query variations, retrieve for each, merge with RRF.
    """
    # Generate query variations
    multi_query_prompt = f"""Generate {num_queries} different versions of this question from different perspectives:

Question: {query}

Variations (one per line):"""

    response = llm.invoke(multi_query_prompt)
    query_variations = [q.strip() for q in response.content.split("\n") if q.strip()][:num_queries]

    # Add original query
    all_queries = [query] + query_variations

    # Retrieve for each query
    all_rankings = []
    for q in all_queries:
        q_embedding = model.encode(q)
        doc_embeddings = model.encode(documents)
        similarities = np.dot(doc_embeddings, q_embedding)
        ranked_indices = np.argsort(similarities)[::-1][:50].tolist()
        all_rankings.append(ranked_indices)

    # Merge with Reciprocal Rank Fusion
    scores = {}
    for ranking in all_rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    # Sort by combined score
    top_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
    results = [documents[i] for i in top_indices]

    return results

# Query: "Explain RAG architecture"
# Variations:
# - "What are the components of RAG systems?"
# - "How does retrieval-augmented generation work?"
# - "Describe the structure of RAG pipelines"
```

**Why this works**: Single queries miss documents that use different terminology or framing. Multiple perspectives increase recall — if one query misses, another catches it. RRF fusion promotes documents that rank high across multiple queries (consensus signal).

**When multi-query helps**: Ambiguous queries that can be interpreted multiple ways. Domain-specific queries where terminology varies (medical, legal). Users who phrase queries poorly.

**When multi-query hurts**: Adds 1-2 seconds LLM latency + multiple retrievals (3-5× slower). Over-diversification dilutes the original intent. Simple queries where variations add noise.

#### Step-back prompting

Generate a more general, abstract version of the query to retrieve foundational context alongside specific details.

```python
def step_back_retrieval(query: str, documents: list[str], k: int = 10):
    """
    Generate a step-back (more general) query, retrieve for both, combine results.
    """
    # Generate step-back query
    step_back_prompt = f"""Given this specific question: "{query}"

Generate a more general, foundational question that would help answer it.

Specific: "What is the time complexity of HNSW search?"
General: "How do approximate nearest neighbor algorithms work?"

General question:"""

    step_back_query = llm.invoke(step_back_prompt).content.strip()

    # Retrieve for both queries
    specific_results = hybrid_search(query, documents, k=k)
    general_results = hybrid_search(step_back_query, documents, k=k)

    # Combine: 70% specific, 30% general
    combined = specific_results[:int(k * 0.7)] + general_results[:int(k * 0.3)]

    # Deduplicate, rerank
    unique = list({doc['chunk_id']: doc for doc in combined}.values())
    reranked = rerank(query, unique, k=k)  # Rerank with original query

    return reranked

# Query: "What are HIPAA's data breach notification requirements?"
# Step-back: "What are the key components of HIPAA regulations?"
# Retrieves: specific breach rules + general HIPAA context
```

**Why this works**: Specific queries retrieve specific details but miss broader context needed to understand them. Step-back queries retrieve foundational information that helps the LLM contextualize specific answers. Reduces hallucination by providing background knowledge.

**When step-back helps**: Complex domain questions where understanding requires background (legal, medical, technical). Users asking about edge cases without understanding the base case. Multi-hop reasoning where intermediate concepts matter.

**When step-back hurts**: Over-generalization dilutes relevance. Simple factual queries ("What is the capital of France?") don't need background. Adds LLM latency (500ms-1s).

---

## Result optimization

Query optimization transforms the query before retrieval. Result optimization processes retrieved chunks to improve diversity, reduce redundancy, or expand context. Three techniques address different quality issues: MMR for diversity, contextual compression for precision, and parent-child retrieval for context.

### MMR (Maximal Marginal Relevance)

Standard retrieval returns the top-k most similar chunks. If you retrieve 5 chunks and all discuss the same narrow aspect, the LLM gets a one-dimensional view. MMR balances relevance and diversity — it selects chunks that are both relevant to the query and different from each other.

**Algorithm**: Start with the most relevant chunk. For each subsequent selection, choose the chunk that maximizes:

```
MMR = λ × sim(query, chunk) - (1 - λ) × max(sim(chunk, selected_chunks))
```

Where:
- `λ` (lambda) controls the relevance vs diversity trade-off
- `sim(query, chunk)` is cosine similarity between query and candidate chunk embeddings
- `max(sim(chunk, selected_chunks))` is the maximum similarity between the candidate and any already-selected chunk

**Implementation**:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

def mmr_rerank(query: str, candidates: list[str], k: int = 5, lambda_param: float = 0.5):
    """
    MMR reranking for diverse results.

    Args:
        query: Search query
        candidates: Retrieved candidate chunks (typically 20-50)
        k: Number of final results
        lambda_param: Relevance weight (0=diversity only, 1=relevance only)

    Returns:
        List of k diverse, relevant chunks
    """
    # Embed query and candidates
    query_embedding = model.encode(query)
    candidate_embeddings = model.encode(candidates)

    # Compute relevance scores (query-chunk similarity)
    relevance_scores = np.dot(candidate_embeddings, query_embedding)

    # Select first chunk (highest relevance)
    selected_indices = [np.argmax(relevance_scores)]
    selected_embeddings = [candidate_embeddings[selected_indices[0]]]

    # Iteratively select k-1 more chunks
    while len(selected_indices) < k:
        # Compute diversity penalty for each remaining candidate
        remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]

        mmr_scores = []
        for i in remaining_indices:
            # Relevance component
            relevance = relevance_scores[i]

            # Diversity component: max similarity to any selected chunk
            similarities = [np.dot(candidate_embeddings[i], emb) for emb in selected_embeddings]
            max_similarity = max(similarities) if similarities else 0

            # MMR formula
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((i, mmr))

        # Select chunk with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        selected_embeddings.append(candidate_embeddings[best_idx])

    # Return selected chunks in order
    return [candidates[i] for i in selected_indices]

# Usage: retrieve 50 candidates, rerank to 5 with MMR
candidates = hybrid_search(query, documents, k=50)
diverse_results = mmr_rerank(query, candidates, k=5, lambda_param=0.5)
```

**Tuning lambda**:
- **λ = 1.0**: Pure relevance, no diversity (equivalent to standard retrieval)
- **λ = 0.7-0.9**: Favor relevance, some diversity (precision search where users know what they want)
- **λ = 0.5**: Balanced (default starting point for most queries)
- **λ = 0.3-0.5**: Favor diversity (exploratory search, multi-faceted queries)
- **λ = 0.0**: Pure diversity, ignore relevance (rarely useful)

**When MMR helps**: Multi-faceted queries where users need diverse perspectives. "Best vacation spots" should return beaches, mountains, cities — not 5 beach recommendations. Summarization tasks where non-redundant information adds value. Questions that benefit from multiple angles ("What are the pros and cons of X?").

**When MMR hurts**: Precision search where redundancy is actually confirmation ("Find HIPAA compliance requirements" — seeing the same requirement in 3 chunks confirms it's important). Latency-critical applications (MMR adds 50-200ms for pairwise similarity computations). Very small result sets (k < 5) where diversity matters less than precision.

**Production considerations**:
- Computational cost scales with O(k²) due to pairwise similarities. Limit reranking to 20-50 candidates.
- Requires document embeddings to be available (not just IDs and text). Store embeddings in your vector database.
- Lambda tuning is domain-specific. Start at 0.5, adjust based on user feedback or A/B testing.
- Apply MMR after stage 2 reranking, not before. Rerank 100 → 50 with cross-encoder, then MMR 50 → 5.

### Contextual compression

Retrieval returns chunks, but not every sentence in a chunk is relevant to the query. Contextual compression filters retrieved chunks to keep only query-relevant content. This reduces LLM context window usage, improves precision, and can allow more documents to fit within token limits.

**Approach 1: LLM-based extraction** (high quality, high cost)

Use an LLM to extract only relevant content from each retrieved chunk.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

def llm_compress(query: str, chunks: list[str]) -> list[str]:
    """
    Use LLM to extract query-relevant content from chunks.

    Compression ratio: typically 4-15x (10,000 tokens → 500-2500 tokens)
    Latency: +100-500ms per chunk (can batch for efficiency)
    """
    compression_prompt = PromptTemplate.from_template("""
Given a query and a text chunk, extract ONLY the sentences that are directly relevant to answering the query.
If no sentences are relevant, return an empty string.

Query: {query}

Chunk:
{chunk}

Relevant sentences (verbatim, no additions):""")

    compressed_chunks = []
    for chunk in chunks:
        response = llm.invoke(compression_prompt.format(query=query, chunk=chunk))
        compressed = response.content.strip()

        # Only include non-empty compressed chunks
        if compressed:
            compressed_chunks.append(compressed)

    return compressed_chunks

# Retrieve 10 chunks, compress to query-relevant content
candidates = hybrid_search(query, documents, k=10)
compressed = llm_compress(query, candidates)
# compressed may have 5-7 chunks (3 filtered as irrelevant), each 20-30% of original length
```

**Cost analysis**: If you retrieve 10 chunks × 500 tokens each = 5000 input tokens, plus 10 LLM calls. At GPT-4 prices (~$30/1M input tokens), that's $0.15 per query. High for production scale. Consider cheaper models (GPT-3.5, Claude Haiku) for compression.

**Approach 2: Embedding-based filtering** (fast, lower cost)

Re-embed individual sentences or paragraphs, filter by similarity threshold.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

def embedding_compress(query: str, chunks: list[str], similarity_threshold: float = 0.5) -> list[str]:
    """
    Filter chunks by embedding similarity to query.

    Compression ratio: 2-5x (depends on threshold)
    Latency: +10-50ms (embedding computation)
    """
    query_embedding = model.encode(query)

    compressed_chunks = []
    for chunk in chunks:
        # Split chunk into sentences (or use whole chunk)
        sentences = chunk.split('. ')

        # Embed sentences
        sentence_embeddings = model.encode(sentences)

        # Compute similarities
        similarities = np.dot(sentence_embeddings, query_embedding)

        # Keep sentences above threshold
        relevant_sentences = [
            sent for sent, sim in zip(sentences, similarities)
            if sim >= similarity_threshold
        ]

        if relevant_sentences:
            compressed_chunks.append('. '.join(relevant_sentences))

    return compressed_chunks

# Retrieve and compress
candidates = hybrid_search(query, documents, k=10)
compressed = embedding_compress(query, candidates, similarity_threshold=0.5)
```

**Tuning similarity threshold**:
- **threshold = 0.3**: Permissive, keeps most content (2x compression)
- **threshold = 0.5**: Balanced (3-4x compression)
- **threshold = 0.7**: Aggressive, only highly relevant sentences (5-10x compression)

Higher thresholds risk removing important context. Lower thresholds keep too much irrelevant content. Start at 0.5.

**Approach 3: Pipeline** (hybrid, optimal quality/cost)

Combine multiple filtering steps: split into sentences → remove redundant sentences → filter by relevance threshold.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

def pipeline_compress(query: str, chunks: list[str], redundancy_threshold: float = 0.85, relevance_threshold: float = 0.5):
    """
    Multi-stage compression pipeline.

    Stage 1: Split into sentences
    Stage 2: Remove redundant sentences (high inter-sentence similarity)
    Stage 3: Filter by query relevance
    """
    query_embedding = model.encode(query)

    # Stage 1: Split all chunks into sentences
    all_sentences = []
    for chunk in chunks:
        sentences = [s.strip() for s in chunk.split('. ') if s.strip()]
        all_sentences.extend(sentences)

    # Stage 2: Remove redundant sentences
    sentence_embeddings = model.encode(all_sentences)

    unique_sentences = []
    unique_embeddings = []

    for sent, emb in zip(all_sentences, sentence_embeddings):
        # Check if similar sentence already selected
        if unique_embeddings:
            similarities = np.dot(unique_embeddings, emb)
            max_sim = np.max(similarities)

            if max_sim < redundancy_threshold:
                unique_sentences.append(sent)
                unique_embeddings.append(emb)
        else:
            unique_sentences.append(sent)
            unique_embeddings.append(emb)

    # Stage 3: Filter by relevance to query
    relevance_scores = np.dot(unique_embeddings, query_embedding)

    compressed = [
        sent for sent, score in zip(unique_sentences, relevance_scores)
        if score >= relevance_threshold
    ]

    return compressed

# Retrieve 15 chunks, compress to ~3-5 chunks worth of content
candidates = hybrid_search(query, documents, k=15)
compressed = pipeline_compress(query, candidates, redundancy_threshold=0.85, relevance_threshold=0.5)
# Returns list of individual relevant, non-redundant sentences
```

**When compression helps**: Context window limits are a concern (long queries, many retrievals). You want to maximize number of documents within token budget. Retrieved chunks contain significant irrelevant content (high recall retrieval). Cost optimization (fewer tokens sent to LLM).

**When compression hurts**: Chunks are already highly relevant and concise (aggressive reranking already applied). Simple queries where full chunks provide needed context. Latency is critical and you can't afford LLM calls. Risk of removing important context (compression too aggressive).

**Production recommendations**:
- Start with embedding-based filtering for fast, cost-effective compression (10-50ms, 2-4x reduction)
- Use LLM-based extraction only when precision is critical and latency is acceptable (100-500ms, 4-15x reduction)
- Apply compression after reranking, not before (rerank first to reduce compression load)
- Monitor compression ratio vs. answer quality in production
- Consider adaptive compression: simple queries → aggressive compression, complex queries → light compression

### Parent-child retrieval

Standard chunking faces a trade-off: small chunks (sentences) match queries precisely but lack context. Large chunks (paragraphs) provide context but match queries poorly. Parent-child retrieval solves this: retrieve based on small chunks (children), return large chunks (parents) to the LLM.

Two implementations: **sentence window retrieval** (fixed expansion) and **auto-merging retrieval** (hierarchical, selective expansion).

#### Sentence window retrieval

Embed individual sentences for precise matching. When a sentence is retrieved, expand to a fixed window of surrounding sentences before passing to the LLM.

**Default configuration**:
- Embed sentences individually
- Store sentence position and parent document metadata
- When retrieved, expand to ±5 sentences (11 total: 5 before, target, 5 after)

**Implementation**:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

class SentenceWindowRetriever:
    """Retrieve sentences, return expanded windows."""

    def __init__(self, documents: list[str], window_size: int = 5):
        """
        Args:
            documents: List of full documents
            window_size: Number of sentences on each side of retrieved sentence
        """
        self.documents = documents
        self.window_size = window_size

        # Split documents into sentences, track position
        self.sentences = []
        self.sentence_metadata = []  # (doc_idx, sentence_idx, total_sentences)

        for doc_idx, doc in enumerate(documents):
            doc_sentences = [s.strip() + '.' for s in doc.split('.') if s.strip()]
            for sent_idx, sentence in enumerate(doc_sentences):
                self.sentences.append(sentence)
                self.sentence_metadata.append({
                    'doc_idx': doc_idx,
                    'sent_idx': sent_idx,
                    'total_sentences': len(doc_sentences),
                    'doc_sentences': doc_sentences,  # Store for window expansion
                })

        # Embed all sentences
        self.sentence_embeddings = model.encode(self.sentences)

    def search(self, query: str, k: int = 5):
        """
        Search for sentences, return expanded windows.

        Args:
            query: Search query
            k: Number of sentence windows to return

        Returns:
            List of expanded sentence windows (each ~11 sentences)
        """
        # Retrieve top-k sentences
        query_embedding = model.encode(query)
        similarities = np.dot(self.sentence_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        # Expand to windows
        windows = []
        for idx in top_indices:
            metadata = self.sentence_metadata[idx]
            doc_sentences = metadata['doc_sentences']
            sent_idx = metadata['sent_idx']

            # Compute window boundaries
            start = max(0, sent_idx - self.window_size)
            end = min(len(doc_sentences), sent_idx + self.window_size + 1)

            # Extract window
            window = ' '.join(doc_sentences[start:end])

            windows.append({
                'window': window,
                'doc_idx': metadata['doc_idx'],
                'target_sentence': self.sentences[idx],
                'window_size': end - start,
            })

        return windows

# Usage
retriever = SentenceWindowRetriever(documents, window_size=5)
results = retriever.search("What are HIPAA's privacy rules?", k=5)

# Each result contains ~11 sentences (5 + target + 5) with the target sentence matched precisely
for i, result in enumerate(results):
    print(f"{i+1}. Target: {result['target_sentence'][:80]}...")
    print(f"   Window: {result['window'][:150]}...")
```

**Tuning window size**:
- **window_size = 1**: Minimal context (3 sentences total)
- **window_size = 3**: Substantial improvement, moderate tokens
- **window_size = 5**: Default balanced setting (11 sentences)
- **window_size > 5**: Diminishing returns, risk of context overload

Empirical results show window_size=3 gives significant quality improvement with reasonable token usage. Window_size=5 is a safe default.

**When sentence window works**: Documents with linear structure (articles, blog posts, documentation). Questions requiring local adjacent context ("What happens after X?" retrieves sentence about X, window includes following sentences). Simple implementation with clear behavior.

**When sentence window fails**: Structured documents where relevant context isn't adjacent (technical docs with cross-references). Token budget is tight (always expands, even when unnecessary). Hierarchical relationships matter (sections, subsections).

#### Auto-merging retrieval

Build a hierarchical tree: parent nodes (large chunks), intermediate nodes (medium chunks), leaf nodes (small chunks). Retrieve leaf nodes via similarity search. If many leaf nodes under the same parent are retrieved, return the parent instead (selective merging).

**Typical configuration**:
- Leaf chunks: 128-256 tokens (sentences, specific concepts)
- Intermediate chunks: 512 tokens (paragraphs)
- Parent chunks: 2048 tokens (sections, multiple paragraphs)
- Merge threshold: 0.6 (if 60% of children retrieved, return parent)

**Implementation** (conceptual, full implementation with LlamaIndex is 100+ lines):

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HierarchicalNode:
    """Node in hierarchical chunk tree."""
    chunk_id: str
    text: str
    level: int  # 0=parent, 1=intermediate, 2=leaf
    parent_id: Optional[str]
    children_ids: List[str]

def build_hierarchical_index(document: str, chunk_sizes: list[int] = [2048, 512, 128]):
    """
    Build hierarchical index with multiple chunk sizes.

    Args:
        document: Full document text
        chunk_sizes: Sizes for [parent, intermediate, leaf] chunks

    Returns:
        Dict of chunk_id -> HierarchicalNode
    """
    nodes = {}

    # Level 0: Parent chunks (2048 tokens)
    parent_chunks = split_document(document, chunk_size=chunk_sizes[0], overlap=100)
    for i, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{i}"

        # Level 1: Intermediate chunks (512 tokens) within parent
        intermediate_chunks = split_document(parent_text, chunk_size=chunk_sizes[1], overlap=50)
        intermediate_ids = []

        for j, intermediate_text in enumerate(intermediate_chunks):
            intermediate_id = f"intermediate_{i}_{j}"
            intermediate_ids.append(intermediate_id)

            # Level 2: Leaf chunks (128 tokens) within intermediate
            leaf_chunks = split_document(intermediate_text, chunk_size=chunk_sizes[2], overlap=20)
            leaf_ids = []

            for k, leaf_text in enumerate(leaf_chunks):
                leaf_id = f"leaf_{i}_{j}_{k}"
                leaf_ids.append(leaf_id)

                # Create leaf node
                nodes[leaf_id] = HierarchicalNode(
                    chunk_id=leaf_id,
                    text=leaf_text,
                    level=2,
                    parent_id=intermediate_id,
                    children_ids=[],
                )

            # Create intermediate node
            nodes[intermediate_id] = HierarchicalNode(
                chunk_id=intermediate_id,
                text=intermediate_text,
                level=1,
                parent_id=parent_id,
                children_ids=leaf_ids,
            )

        # Create parent node
        nodes[parent_id] = HierarchicalNode(
            chunk_id=parent_id,
            text=parent_text,
            level=0,
            parent_id=None,
            children_ids=intermediate_ids,
        )

    return nodes

def auto_merge_retrieve(query: str, nodes: dict, k: int = 12, merge_threshold: float = 0.6):
    """
    Retrieve leaf nodes, merge to parents if threshold exceeded.

    Args:
        query: Search query
        nodes: Dict of chunk_id -> HierarchicalNode
        k: Number of leaf nodes to retrieve (larger than final return count)
        merge_threshold: Fraction of children needed to return parent

    Returns:
        List of merged chunks (parents where merging occurred, else leaves)
    """
    # Retrieve top-k leaf nodes
    leaf_nodes = [node for node in nodes.values() if node.level == 2]
    leaf_texts = [node.text for node in leaf_nodes]
    leaf_embeddings = model.encode(leaf_texts)

    query_embedding = model.encode(query)
    similarities = np.dot(leaf_embeddings, query_embedding)

    top_indices = np.argsort(similarities)[::-1][:k]
    retrieved_leaves = [leaf_nodes[i] for i in top_indices]

    # Check for merging opportunities
    parent_to_children = {}
    for leaf in retrieved_leaves:
        parent_id = leaf.parent_id
        if parent_id not in parent_to_children:
            parent_to_children[parent_id] = []
        parent_to_children[parent_id].append(leaf.chunk_id)

    # Merge if threshold exceeded
    final_chunks = []
    for parent_id, retrieved_child_ids in parent_to_children.items():
        parent = nodes[parent_id]

        # Calculate fraction of children retrieved
        fraction = len(retrieved_child_ids) / len(parent.children_ids)

        if fraction >= merge_threshold:
            # Return parent (merged)
            final_chunks.append(parent.text)
        else:
            # Return individual children (no merge)
            for child_id in retrieved_child_ids:
                final_chunks.append(nodes[child_id].text)

    return final_chunks

# Usage (simplified)
nodes = build_hierarchical_index(document, chunk_sizes=[2048, 512, 128])
results = auto_merge_retrieve(query, nodes, k=12, merge_threshold=0.6)
# Returns merged parent chunks where 60%+ children retrieved, else leaf chunks
```

**Tuning merge threshold**:
- **threshold = 0.5**: Aggressive merging (merge if half of children retrieved)
- **threshold = 0.6**: Balanced default
- **threshold = 0.7-0.8**: Conservative merging (need most children before merging)

Lower thresholds increase parent returns (more context, more tokens). Higher thresholds favor leaf precision.

**When auto-merging works**: Hierarchically structured documents (technical docs, research papers, legal documents). Long-range dependencies where context spans sections. Want token efficiency (only expand when query requires broader context). Complex multi-section documents.

**When auto-merging fails**: Simple linear documents where sentence window suffices. Implementation complexity not justified. Small documents where hierarchy doesn't help. Latency is critical (merge logic + larger k for leaf retrieval adds overhead).

**Comparison: Sentence window vs auto-merging**:

| Aspect | Sentence Window | Auto-Merging |
|--------|-----------------|--------------|
| **Expansion strategy** | Fixed (always expands) | Selective (threshold-based) |
| **Context type** | Local (adjacent sentences) | Hierarchical (structured sections) |
| **Implementation** | Simple | Complex |
| **Token efficiency** | Lower (always expanded) | Higher (conditional) |
| **Best for** | Linear documents | Structured documents |
| **Latency** | Low | Medium-high |

**Production recommendations**:
- Start with sentence window for linear content (articles, blogs), window_size=3-5
- Use auto-merging for structured documents (technical docs, research papers), threshold=0.6
- Combine with reranking: retrieve with parent-child, then rerank final chunks
- Monitor token usage vs. answer quality
- Tune window_size or merge_threshold based on domain (legal docs need more context, tweets need less)

### Document-level vs chunk-level retrieval

Chunking is the default for RAG, but sometimes you should return entire documents instead of chunks. Short documents, single-purpose content, and queries requiring narrative coherence benefit from document-level retrieval.

**When to skip chunking entirely**:

1. **Short documents** (< 500 tokens for older models, < 4K-8K for modern LLMs): If the document fits in the LLM context window, don't chunk. Pass the entire document.

```python
def should_chunk(document: str, max_context_tokens: int = 4000) -> bool:
    """Decide whether to chunk based on document length."""
    token_count = len(document.split())  # Rough estimate

    if token_count < max_context_tokens * 0.8:  # 80% of context window
        return False  # Skip chunking, use full document

    return True  # Chunk as normal

# Filter during indexing
for doc in documents:
    if should_chunk(doc.text, max_context_tokens=4000):
        chunks = chunk_document(doc.text)
        index_chunks(chunks)
    else:
        # Index entire document as single unit
        index_document(doc.text, is_full_document=True)
```

2. **Single-purpose documents**: FAQs, product descriptions, short policies, or any focused content where chunking fragments meaning.

3. **Narrative flow matters**: Documents where understanding requires reading start-to-finish (short stories, legal clauses, executive summaries).

**Hybrid approach: Parent document retrieval**

Index small chunks for precise matching, but retrieve and return larger parent chunks or full documents to the LLM. This balances precision (small chunk similarity) with context (large chunk or full document comprehension).

**Implementation with LangChain**:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Child splitter: small chunks for indexing (precise matching)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Small for precision
    chunk_overlap=50
)

# Parent splitter: larger chunks for retrieval (context)
# Omit parent_splitter to return full documents instead
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Large for context
    chunk_overlap=200
)

# Storage for parent documents/chunks
docstore = InMemoryStore()

# Create retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,  # Omit for full document retrieval
)

# Add documents
retriever.add_documents(documents)

# At retrieval time:
# 1. Query embeds and searches child chunks (400 tokens) in vector store
# 2. Retrieved child chunks map to parent IDs
# 3. Parent chunks (2000 tokens) or full documents returned from docstore
retrieved_docs = retriever.get_relevant_documents("What are the key privacy requirements?")
# Returns parent chunks (2000 tokens), not child chunks (400 tokens)
```

**Two modes**:
- **Mode 1 (Full Document)**: Index small chunks, retrieve full documents
  - Use when documents are <5000 tokens and coherence matters
  - Example: FAQs, short articles, product specs
- **Mode 2 (Larger Chunks)**: Index small chunks, retrieve larger chunks
  - Use when documents are >5000 tokens and full document is too large
  - Example: Technical docs, research papers, long reports

**Query characteristics indicating full document retrieval**:

```python
def query_needs_full_document(query: str) -> bool:
    """
    Heuristics to detect queries that need full document context.

    Indicators:
    - Summarization requests
    - Questions about document metadata
    - Queries requiring understanding of overall structure/argument
    """
    summarization_keywords = ['summarize', 'overview', 'key points', 'main ideas', 'summary']
    metadata_keywords = ['author', 'date', 'published', 'who wrote', 'when was']
    structural_keywords = ['argument', 'thesis', 'conclusion', 'overall', 'entire']

    query_lower = query.lower()

    # Check for summarization
    if any(kw in query_lower for kw in summarization_keywords):
        return True

    # Check for metadata questions
    if any(kw in query_lower for kw in metadata_keywords):
        return True

    # Check for structural understanding
    if any(kw in query_lower for kw in structural_keywords):
        return True

    return False

# At retrieval time, choose strategy
if query_needs_full_document(query):
    # Use parent document retriever or query document-level index
    results = parent_doc_retriever.get_relevant_documents(query)
else:
    # Use standard chunk-level retrieval
    results = chunk_retriever.get_relevant_documents(query)
```

**Trade-offs**:

| Aspect | Chunk-Level | Document-Level | Parent Document |
|--------|-------------|----------------|-----------------|
| **Precision** | High | Low | High (child chunks) |
| **Context** | Low | High | High (parent chunks) |
| **Token Cost** | Low | High | Medium-High |
| **Storage** | Low | Low | High (child + parent) |
| **Latency** | Low | Low | Medium (extra lookup) |
| **Complexity** | Low | Low | Medium |

**When document-level retrieval helps**:
- Short documents (<500 tokens for simple tasks, <4K for modern LLMs)
- Single-purpose focused content
- Queries requiring narrative coherence
- Summarization or overview requests
- Documents where chunking loses critical relationships

**When document-level retrieval hurts**:
- Large documents (>5000 tokens) — wastes context window on irrelevant sections
- Precise factual queries — chunks provide better focus
- Multi-topic documents — single vector doesn't capture diversity
- Cost-sensitive applications — full documents burn tokens fast

**Production recommendations**:
- Set document size threshold: <4000 tokens → no chunking, 4000-10000 → parent document retrieval, >10000 → standard chunking
- Use query classification to route: summarization queries → document-level, specific questions → chunk-level
- Monitor token usage: if 80% of LLM context is unused, you're retrieving too much (over-fetching)
- Combine strategies: small documents as-is, large documents chunked, both in same index with `is_full_document` flag

---

## Workflow: building a retrieval pipeline

Retrieval is not a single search call. It's a pipeline: query processing → stage 1 retrieval → stage 2 reranking → result validation. This section walks through building retrieval pipelines from simple to production-ready.

### Simple dense retrieval

Start with the simplest possible retrieval: embed the query, search the vector index, return top-k.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model and index (built during indexing phase)
model = SentenceTransformer('all-mpnet-base-v2')
index = faiss.read_index("index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

def search(query: str, k: int = 5):
    """Simple dense retrieval."""
    # Embed query
    query_embedding = model.encode(query)

    # Search index
    distances, indices = index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        k=k
    )

    # Return results
    results = [
        {"chunk": chunks[i], "score": float(distances[0][j])}
        for j, i in enumerate(indices[0])
    ]

    return results

# Usage
results = search("What is RAG?", k=5)
for i, result in enumerate(results):
    print(f"{i+1}. {result['chunk'][:100]}... (score: {result['score']:.3f})")
```

**What this gives you**: Working retrieval in 20 lines. Good enough for prototypes and validating that RAG helps.

**Limitations**: No keyword matching (fails on "HIPAA compliance"), no reranking (precision limited by stage 1), no error handling.

**When to use**: Prototypes, demos, proof-of-concepts where 80% accuracy is acceptable.

### Two-stage pipeline

Add cross-encoder reranking to improve precision without sacrificing recall.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

# Load models
retriever = SentenceTransformer('all-mpnet-base-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
index = faiss.read_index("index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

def two_stage_search(query: str, stage1_k: int = 50, final_k: int = 5):
    """
    Two-stage retrieval: fast stage 1 → accurate stage 2.

    Stage 1: Retrieve 50 candidates with two-tower
    Stage 2: Rerank to top 5 with cross-encoder
    """
    # Stage 1: Fast retrieval (50 candidates)
    query_embedding = retriever.encode(query)
    distances, indices = index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        k=stage1_k
    )

    stage1_chunks = [chunks[i] for i in indices[0]]

    # Stage 2: Rerank with cross-encoder
    pairs = [[query, chunk] for chunk in stage1_chunks]
    rerank_scores = reranker.predict(pairs)

    # Sort by reranking score
    reranked_indices = np.argsort(rerank_scores)[::-1]

    # Return top-k after reranking
    results = [
        {
            "chunk": stage1_chunks[i],
            "rerank_score": float(rerank_scores[i]),
            "stage1_rank": int(i + 1),
        }
        for i in reranked_indices[:final_k]
    ]

    return results

# Usage
results = two_stage_search("What are HIPAA's privacy rules?", stage1_k=50, final_k=5)
for i, result in enumerate(results):
    print(f"{i+1}. {result['chunk'][:80]}...")
    print(f"   Rerank score: {result['rerank_score']:.3f}, Stage 1 rank: {result['stage1_rank']}")
```

**What this adds**:
- **Better precision**: Reranker improves MRR by 10-20%
- **Recovers from stage 1 errors**: Correct chunk ranked 20th in stage 1 can reach top 5 after reranking
- **Latency budget**: Stage 1 (30-50ms) + Stage 2 (30-50ms) = 60-100ms total

**When to use**: Production systems where precision matters. When stage 1 recall is high (>90%) but precision is low (<70%).

### Hybrid with reranking

Add BM25 sparse retrieval to handle keyword queries and named entities.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

class HybridRetriever:
    """Hybrid search combining dense + sparse retrieval with reranking."""

    def __init__(self, index_path: str, chunks_path: str):
        # Load dense retrieval components
        self.retriever = SentenceTransformer('all-mpnet-base-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = faiss.read_index(index_path)
        self.chunks = np.load(chunks_path, allow_pickle=True)

        # Build BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, stage1_k: int = 100, final_k: int = 5, alpha: float = 0.5):
        """
        Hybrid search with reranking.

        Args:
            query: Search query
            stage1_k: Number of candidates from stage 1
            final_k: Number of final results after reranking
            alpha: Dense weight (0=sparse only, 1=dense only, 0.5=balanced)
        """
        # Dense retrieval
        query_embedding = self.retriever.encode(query)
        dense_distances, dense_indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k=stage1_k
        )
        dense_scores = 1 / (1 + dense_distances[0])  # Convert distance to similarity

        # Sparse retrieval
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

        # Combine scores with alpha weighting
        hybrid_scores = np.zeros(len(self.chunks))
        for i, idx in enumerate(dense_indices[0]):
            hybrid_scores[idx] = alpha * dense_norm[i]
        hybrid_scores += (1 - alpha) * sparse_norm

        # Get top stage1_k candidates
        top_indices = np.argsort(hybrid_scores)[::-1][:stage1_k]
        stage1_chunks = [self.chunks[i] for i in top_indices]

        # Rerank with cross-encoder
        pairs = [[query, chunk] for chunk in stage1_chunks]
        rerank_scores = self.reranker.predict(pairs)

        # Get final top-k
        reranked_indices = np.argsort(rerank_scores)[::-1][:final_k]

        results = [
            {
                "chunk": stage1_chunks[i],
                "rerank_score": float(rerank_scores[i]),
                "hybrid_score": float(hybrid_scores[top_indices[i]]),
                "chunk_id": int(top_indices[i]),
            }
            for i in reranked_indices
        ]

        return results

# Usage
retriever = HybridRetriever("index.faiss", "chunks.npy")

# Semantic query (favor dense)
results = retriever.search("Explain privacy regulations", alpha=0.7, final_k=5)

# Keyword query (favor sparse)
results = retriever.search("HIPAA compliance", alpha=0.3, final_k=5)

# Balanced
results = retriever.search("What are HIPAA's privacy rules?", alpha=0.5, final_k=5)
```

**What this adds**:
- **Keyword matching**: BM25 catches exact terms (acronyms, named entities)
- **Tunable balance**: Adjust alpha per query type
- **Best of both worlds**: 5-15% Hit Rate improvement over dense-only or sparse-only

**When to use**: Production systems with diverse query types. When users search for named entities, acronyms, or exact phrases alongside semantic queries.

### Production pipeline with monitoring

Add error handling, logging, caching, and metrics tracking for production deployment.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import time
import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval request."""
    query: str
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms: float
    stage1_count: int
    final_count: int
    alpha: float

class ProductionRetriever:
    """Production retrieval pipeline with monitoring and error handling."""

    def __init__(self, index_path: str, chunks_path: str, cache_size: int = 1000):
        self.retriever = SentenceTransformer('all-mpnet-base-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = faiss.read_index(index_path)
        self.chunks = np.load(chunks_path, allow_pickle=True)

        # Build BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # Metrics storage
        self.metrics_history: List[RetrievalMetrics] = []

        logger.info(f"Initialized retriever with {len(self.chunks)} chunks")

    @lru_cache(maxsize=1000)
    def _embed_query_cached(self, query: str):
        """Cache query embeddings for repeated queries."""
        return tuple(self.retriever.encode(query).tolist())

    def search(
        self,
        query: str,
        stage1_k: int = 100,
        final_k: int = 5,
        alpha: float = 0.5,
        enable_cache: bool = True,
    ) -> Dict:
        """
        Production retrieval with monitoring.

        Returns:
            Dict with keys: results, metrics
        """
        start_time = time.time()

        try:
            # Stage 1: Hybrid retrieval
            stage1_start = time.time()

            # Dense retrieval (with caching)
            if enable_cache:
                query_embedding = np.array(self._embed_query_cached(query))
            else:
                query_embedding = self.retriever.encode(query)

            dense_distances, dense_indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k=stage1_k
            )
            dense_scores = 1 / (1 + dense_distances[0])

            # Sparse retrieval
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)

            # Combine
            dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
            sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

            hybrid_scores = np.zeros(len(self.chunks))
            for i, idx in enumerate(dense_indices[0]):
                hybrid_scores[idx] = alpha * dense_norm[i]
            hybrid_scores += (1 - alpha) * sparse_norm

            top_indices = np.argsort(hybrid_scores)[::-1][:stage1_k]
            stage1_chunks = [self.chunks[i] for i in top_indices]

            stage1_latency = (time.time() - stage1_start) * 1000

            # Stage 2: Reranking
            stage2_start = time.time()

            pairs = [[query, chunk] for chunk in stage1_chunks]
            rerank_scores = self.reranker.predict(pairs)

            reranked_indices = np.argsort(rerank_scores)[::-1][:final_k]

            stage2_latency = (time.time() - stage2_start) * 1000

            # Build results
            results = [
                {
                    "chunk": stage1_chunks[i],
                    "rerank_score": float(rerank_scores[i]),
                    "chunk_id": int(top_indices[i]),
                }
                for i in reranked_indices
            ]

            # Record metrics
            total_latency = (time.time() - start_time) * 1000
            metrics = RetrievalMetrics(
                query=query,
                stage1_latency_ms=stage1_latency,
                stage2_latency_ms=stage2_latency,
                total_latency_ms=total_latency,
                stage1_count=stage1_k,
                final_count=final_k,
                alpha=alpha,
            )
            self.metrics_history.append(metrics)

            logger.info(
                f"Retrieved {final_k} chunks in {total_latency:.1f}ms "
                f"(stage1: {stage1_latency:.1f}ms, stage2: {stage2_latency:.1f}ms)"
            )

            return {
                "results": results,
                "metrics": {
                    "stage1_latency_ms": stage1_latency,
                    "stage2_latency_ms": stage2_latency,
                    "total_latency_ms": total_latency,
                },
            }

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise

    def get_performance_summary(self) -> Dict:
        """Get performance statistics from recent retrievals."""
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-100:]  # Last 100 queries

        return {
            "total_queries": len(recent),
            "avg_total_latency_ms": np.mean([m.total_latency_ms for m in recent]),
            "p95_total_latency_ms": np.percentile([m.total_latency_ms for m in recent], 95),
            "avg_stage1_latency_ms": np.mean([m.stage1_latency_ms for m in recent]),
            "avg_stage2_latency_ms": np.mean([m.stage2_latency_ms for m in recent]),
        }

# Usage
retriever = ProductionRetriever("index.faiss", "chunks.npy")

# Retrieve
response = retriever.search("What are HIPAA's privacy rules?", stage1_k=100, final_k=5)
print(f"Latency: {response['metrics']['total_latency_ms']:.1f}ms")

# Check performance after 100 queries
summary = retriever.get_performance_summary()
print(f"Avg latency: {summary['avg_total_latency_ms']:.1f}ms, P95: {summary['p95_total_latency_ms']:.1f}ms")
```

**What this adds**:
- **Query embedding cache**: 30ms → <1ms for repeated queries (exact-match LRU cache on query string; see semantic caching below for similarity-based caching)
- **Latency tracking**: Monitor stage 1, stage 2, and total latency per query
- **Error handling**: Failed retrievals logged, don't crash the system
- **Performance summary**: P95 latency, average latency per stage

**When to use**: Production systems serving real users. When you need observability and SLAs.

#### Semantic caching

The LRU cache above (line 2099) caches query embeddings for **exact string matches** only — "What are HIPAA rules?" and "What are the HIPAA rules?" miss the cache despite being semantically identical. **Semantic caching** caches retrieval results based on query **similarity** (cosine similarity ≥ 0.90-0.95), handling paraphrased queries and achieving 40-70% hit rates (vs 10-20% for exact-match) in FAQ/support systems.

**Distinguish from existing cache**:
- **Existing (line 2099)**: LRU cache on query **string** → exact match → caches query **embeddings** (30ms → <1ms for encoding)
- **Semantic caching**: Cache query **embeddings + retrieval results** → similarity match (≥0.90-0.95 threshold) → caches full **retrieval pipeline** (100ms → 12-25ms)

**What to cache** (choose based on trade-offs):

| What to cache | Saves | Latency gain | Precision | Use case |
|---------------|-------|--------------|-----------|----------|
| Query embeddings only | Encoding (30ms) | 30ms → <1ms | N/A | Existing LRU (exact-match) |
| Retrieval results (chunks) | Encoding + retrieval + reranking (100ms) | 100ms → 12-25ms | 90-98% | **Recommended**: FAQ, support |
| Full LLM generations | Entire pipeline (2-5s) | 2-5s → 50-100ms | 85-95% | Static content (docs, references) |

**Redis implementation** (most production-ready option):

```python
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict
import hashlib
import json

class SemanticCache:
    """
    Semantic cache for retrieval results using Redis.

    Caches query embeddings + retrieval results. On cache hit (similarity ≥ threshold),
    return cached results. On miss, retrieve, cache, and return.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        encoder: SentenceTransformer,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        self.redis = redis_client
        self.encoder = encoder
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

        # Cache key prefixes
        self.QUERY_PREFIX = "cache:query:"
        self.RESULTS_PREFIX = "cache:results:"
        self.INDEX_KEY = "cache:query_index"  # Set of all cached query IDs

    def _query_id(self, query_embedding: np.ndarray) -> str:
        """Generate stable ID from query embedding."""
        embedding_bytes = query_embedding.astype(np.float32).tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()[:16]

    def get(self, query: str) -> Optional[List[Dict]]:
        """
        Get cached results if similar query exists.

        Returns:
            Cached results if hit (similarity ≥ threshold), None if miss
        """
        # Encode query
        query_embedding = self.encoder.encode(query)

        # Get all cached query embeddings
        cached_query_ids = self.redis.smembers(self.INDEX_KEY)

        if not cached_query_ids:
            return None

        # Compare with all cached queries (brute force for <10K cached queries)
        # For >10K cached queries, use approximate nearest neighbor (FAISS, Qdrant)
        max_similarity = -1.0
        best_match_id = None

        for query_id in cached_query_ids:
            query_id = query_id.decode('utf-8')
            cached_embedding_json = self.redis.get(f"{self.QUERY_PREFIX}{query_id}")

            if cached_embedding_json is None:
                continue  # Cache expired

            cached_embedding = np.array(json.loads(cached_embedding_json))
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = query_id

        # Cache hit if similarity above threshold
        if max_similarity >= self.threshold and best_match_id:
            results_json = self.redis.get(f"{self.RESULTS_PREFIX}{best_match_id}")
            if results_json:
                return json.loads(results_json)

        return None  # Cache miss

    def set(self, query: str, results: List[Dict]) -> None:
        """
        Cache query embedding + retrieval results.

        Args:
            query: Query string
            results: List of retrieved chunks (from ProductionRetriever)
        """
        query_embedding = self.encoder.encode(query)
        query_id = self._query_id(query_embedding)

        # Store query embedding
        self.redis.setex(
            f"{self.QUERY_PREFIX}{query_id}",
            self.ttl,
            json.dumps(query_embedding.tolist())
        )

        # Store results
        self.redis.setex(
            f"{self.RESULTS_PREFIX}{query_id}",
            self.ttl,
            json.dumps(results)
        )

        # Add to index
        self.redis.sadd(self.INDEX_KEY, query_id)


# Usage with ProductionRetriever
redis_client = redis.Redis(host='localhost', port=6379, db=0)
encoder = SentenceTransformer('all-mpnet-base-v2')
cache = SemanticCache(
    redis_client=redis_client,
    encoder=encoder,
    similarity_threshold=0.92,  # 92% similarity required for cache hit
    ttl_seconds=3600,  # 1 hour TTL
)

retriever = ProductionRetriever("index.faiss", "chunks.npy")

def retrieve_with_cache(query: str, stage1_k: int = 100, final_k: int = 5) -> Dict:
    """Retrieval with semantic caching."""
    # Check cache first
    cached_results = cache.get(query)
    if cached_results is not None:
        logger.info(f"Cache HIT for query: {query}")
        return {
            "results": cached_results,
            "from_cache": True,
            "latency_ms": 12,  # Typical cache latency
        }

    # Cache miss: perform retrieval
    logger.info(f"Cache MISS for query: {query}")
    response = retriever.search(query, stage1_k=stage1_k, final_k=final_k)

    # Cache results
    cache.set(query, response["results"])

    return {
        "results": response["results"],
        "from_cache": False,
        "latency_ms": response["metrics"]["total_latency_ms"],
    }


# Example: Paraphrased queries hit cache
response1 = retrieve_with_cache("What are HIPAA's privacy rules?")  # Cache miss (100ms)
response2 = retrieve_with_cache("What are the privacy rules in HIPAA?")  # Cache HIT (12ms) — 92% similar
response3 = retrieve_with_cache("Explain HIPAA privacy requirements")  # Cache HIT (12ms) — 94% similar
```

**Similarity threshold tuning** (trade precision vs hit rate):

| Threshold | Hit Rate | Precision | Use case |
|-----------|----------|-----------|----------|
| 0.85-0.90 | 50-70% | 85-92% | FAQ systems, high tolerance for false positives |
| **0.90-0.95** | **40-60%** | **92-98%** | **Recommended**: Customer support, general RAG |
| 0.95-0.99 | 20-40% | 98-99.5% | Compliance-critical, zero tolerance for incorrect cached results |

**Lower threshold**: More cache hits but higher risk of returning cached results for dissimilar queries (false positives).
**Higher threshold**: Fewer false positives but lower hit rate (more cache misses).

**False positive management** (preventing bad cached results):

1. **Pre-load gold-standard entries**: Seed cache with verified query-answer pairs for common questions (FAQ, documentation queries).

2. **Quality control via sampling**: Log 1-5% of cache hits with `similarity < 0.95` for manual review. If false positive rate >5%, increase threshold.

3. **Cross-encoder validation** (optional): Before returning cached results, run cross-encoder similarity check between cached query and current query. If cross-encoder score <0.8, treat as cache miss.

   ```python
   # Add to SemanticCache.get()
   if max_similarity >= self.threshold and best_match_id:
       # Optional: Cross-encoder validation for borderline hits
       if self.threshold < 0.95:
           cross_encoder_score = self.cross_encoder.predict([(cached_query_text, query)])[0]
           if cross_encoder_score < 0.8:
               return None  # Reject borderline cache hit

       results_json = self.redis.get(f"{self.RESULTS_PREFIX}{best_match_id}")
       if results_json:
           return json.loads(results_json)
   ```

4. **User feedback**: Add "Was this helpful?" to cached results. If negative feedback rate >10%, increase threshold or invalidate that cache entry.

**Cache invalidation strategies**:

| Strategy | TTL | Use case | Trade-off |
|----------|-----|----------|-----------|
| **Time-based (TTL)** | 1 hour | Dynamic content (support tickets, live docs) | Simple but may serve stale results |
| | 24 hours | Stable content (product docs, FAQs) | Balanced |
| | 7 days | Reference content (compliance docs, archives) | Maximum hit rate but staleness risk |
| **Event-driven** | Invalidate on document update | Critical accuracy (legal, financial) | Complex (requires document change tracking) |
| **Hybrid** | TTL + event-driven invalidation | Production RAG | Recommended: Balance simplicity and freshness |

**Performance metrics** (measured on customer support RAG, 10K queries/day):

| Metric | Value | Details |
|--------|-------|---------|
| **Hit rate** | 40-70% | FAQ systems (high repetition), 20-30% for exploratory research |
| **Latency reduction** | 4-8× faster | 100ms → 12-25ms (cache hit includes Redis roundtrip + deserialization) |
| **Cost savings** | 40-73% | Proportional to hit rate (fewer LLM calls, fewer vector DB queries) |
| **Storage** | ~10MB / 1000 cached queries | Query embeddings (768-dim) + JSON results (5-50KB/query) |
| **False positive rate** | 2-8% | At 0.90-0.95 threshold (validated via cross-encoder sampling) |

**When to use semantic caching**:
- **FAQ systems**: High query repetition (users ask same questions in different words)
- **Customer support**: Common troubleshooting queries ("How do I reset password?" vs "Reset password steps")
- **Documentation search**: Queries about established topics (APIs, features, guides)
- **Time-insensitive domains**: Content doesn't change frequently (reference docs, historical data)
- **Cost/latency critical**: 40-70% hit rate = 40-70% cost reduction + 4-8× faster

**When to skip**:
- **Time-sensitive data**: News, stock prices, real-time monitoring (cached results quickly become stale)
- **Low query repetition**: Research, exploration, ad-hoc analysis (hit rate <20% = overhead not justified)
- **Highly personalized**: User-specific context makes caching ineffective (each user's query is unique)
- **Small scale**: <100 queries/day (cache overhead exceeds benefit)

**Alternatives**:
- **GPTCache** (Zilliz): Purpose-built semantic cache with vector DB backend, supports similarity search and LLM response caching
- **LangChain semantic cache**: Built-in support for Redis, SQLite, or in-memory semantic caching
- **Redis Stack**: Adds vector similarity search to Redis for efficient semantic caching at scale (>10K cached queries)

---

## Common pitfalls

**Using only dense retrieval**: Dense retrieval misses exact keyword matches. "HIPAA compliance" should strongly favor documents containing "HIPAA" even if semantic similarity is moderate. Use hybrid search (dense + sparse) to get both semantic and keyword matching.

**Skipping reranking**: Two-tower models are fast but less accurate than cross-encoders. If you have >50ms latency budget, add reranking. It typically improves MRR by 10-20% with minimal added complexity.

**Reranking too few candidates**: If you retrieve 20 candidates in stage 1 and rerank to 5, the reranker has limited options to fix stage 1 errors. Retrieve 50-100 candidates, rerank to 10-20, then return top 5. Give the reranker room to work.

**Not tuning efSearch**: HNSW defaults (efSearch=16) prioritize speed over recall. For RAG, you want high recall in stage 1. Set efSearch=50-100 to get 95-98% recall. The reranker compensates for the extra latency.

**Returning low-scoring results**: If all top-5 reranking scores are <0.3, the query likely has no relevant chunks. Return an empty result or a "no relevant information" message instead of showing irrelevant chunks to the LLM.

**Ignoring query length**: Single-word queries ("RAG") are ambiguous and need query expansion. Long queries (50+ words) are overly specific and need query decomposition. Adapt your pipeline to query complexity.

**Not caching embeddings for common queries**: If 20% of queries repeat ("What is X?", "How do I Y?"), cache their embeddings. Query embedding takes 20-30ms — caching reduces this to <1ms.

**Filtering after retrieval**: If you retrieve 100 chunks then filter by metadata, you're wasting retrieval. Filter at query time using vector database filters: "only search category=healthcare" restricts the search space before retrieval.

**Using identical k for all queries**: Simple queries might only need 3 chunks. Complex queries might need 10. Return variable k based on query complexity or confidence scores.

**Not monitoring retrieval independently**: If the LLM hallucinates, is it because retrieval failed (correct chunk not in top-5) or generation failed (correct chunk present but LLM ignored it)? Track Hit Rate@5 and MRR separately from generation quality.

**Mixing incompatible score scales**: Dense retrieval scores might range [0.3, 0.9], BM25 scores might range [0, 150]. Without normalization, BM25 dominates hybrid search. Always normalize to [0, 1] before combining.

**Over-optimizing stage 2**: If stage 1 recall is 70%, no amount of reranking will fix it. The correct chunk must reach stage 2 to be reranked. Optimize stage 1 for recall first, then optimize stage 2 for precision.

**Applying MMR too early**: Don't apply MMR before reranking. Rerank first with a cross-encoder (100 → 50 candidates), then apply MMR (50 → 5 final results). MMR operates on embeddings — reranking provides better quality signals than stage 1 similarity.

**Over-compressing context**: Aggressive compression (threshold=0.7, LLM extraction) can remove important context that seems irrelevant to a simple relevance check but matters for complex reasoning. Start with light compression (threshold=0.5), monitor answer quality, adjust based on retrieval metrics.

**Using sentence window for structured documents**: Sentence window works for linear content (blog posts, articles) where context is adjacent. For technical docs with cross-references or hierarchical structure (sections, subsections), use auto-merging retrieval instead. Sentence window will miss non-adjacent relevant context.

**Parent-child retrieval without reranking**: Sentence window and auto-merging expand retrieved chunks, increasing token count 5-10×. If you skip reranking, you'll pass low-quality expanded chunks to the LLM. Always apply reranking after expansion to filter the best expanded contexts.

**Inadequate RBAC metadata preventing granular filtering**: If you add only coarse-grained `access_level` metadata (public/internal/confidential), you can't implement user-specific or role-specific filtering. Add hierarchical RBAC fields at indexing time (roles, departments, owner_id) to enable query-time hard security filtering.

**Using ColBERT for <100 candidates**: ColBERT adds 10-50ms latency for reranking. If you only have 50 candidates, cross-encoder is fast enough (50ms) and more accurate. Only use ColBERT when reranking 100-1000 candidates where cross-encoder would be too slow (>100ms).

**Over-aggressive semantic cache threshold (0.95-0.99)**: High thresholds reduce hit rate to 20-30% (barely better than exact-match). Use 0.90-0.95 for 40-60% hit rate. If false positives are a concern, add cross-encoder validation before returning cached results.

**Caching without TTL serves stale results**: Time-based data (product docs, support articles) changes frequently. If you cache without TTL, users see outdated results. Set domain-appropriate TTL: 1 hour (dynamic), 24 hours (stable), 7 days (reference). Add event-driven invalidation for critical accuracy.

---

## Production quality metrics & SLAs

RAG systems need measurable success criteria. Business stakeholders need to know: "What's good enough for production?" and "How do we know if it's working?" This section defines target metrics and SLA benchmarks.

### Retrieval quality benchmarks

**Hit Rate@k**: Fraction of queries where the correct chunk appears in the top-k results.

| Quality Level | Hit Rate@5 | Hit Rate@10 | When Acceptable |
|---------------|-----------|-------------|-----------------|
| **Minimum viable** | >70% | >80% | Early prototype, internal beta testing |
| **Production ready** | >85% | >90% | Customer-facing features, FAQ systems |
| **Best-in-class** | >95% | >98% | High-stakes applications (legal, medical, compliance) |

**Mean Reciprocal Rank (MRR)**: Average of 1/rank where rank is the position of the first correct chunk. Higher is better (1.0 = always rank 1).

| Quality Level | MRR | Interpretation |
|---------------|-----|----------------|
| **Below target** | <0.5 | Correct chunk typically ranked >2nd, retrieval needs improvement |
| **Good** | 0.6-0.8 | Correct chunk typically ranked 1st-2nd, acceptable for most use cases |
| **Excellent** | >0.85 | Correct chunk almost always ranked 1st, best-in-class retrieval |

**How to measure**: Create labeled test set (200-500 queries with ground-truth correct chunks), run retrieval pipeline, compute metrics weekly. Alert if Hit Rate@5 drops <85% or MRR <0.6 (indicates quality regression).

### Latency SLAs

**Retrieval latency targets** (excludes LLM generation):

| Percentile | Target Latency | Use Case |
|-----------|---------------|----------|
| **p50** (median) | <100ms | Interactive RAG, chatbots, search |
| **p95** | <200ms | Acceptable for 95% of users |
| **p99** | <500ms | Edge cases, complex queries |
| **Timeout** | 1000ms (1 second) | Fail gracefully, show error message |

**Latency breakdown by stage**:

| Stage | Target Latency | Notes |
|-------|----------------|-------|
| **Stage 1 (hybrid retrieval)** | 30-50ms | Two-tower + BM25, retrieve 50-100 candidates |
| **Stage 2 (reranking)** | 30-80ms | Cross-encoder, rerank to top 5-10 |
| **Caching lookup** | 5-15ms | Semantic cache hit (if applicable) |
| **Total** | 60-130ms (p50) | Fits within 300ms end-to-end SLA (retrieval + generation) |

**How to optimize**:
- If stage 1 >50ms: Lower efSearch (20-30), use IVF instead of HNSW, reduce candidate count
- If stage 2 >80ms: Use smaller reranker (6-layer vs 12-layer), reduce candidates (50 → 30), skip reranking for simple queries
- If total >200ms: Add semantic caching (40-70% queries serve from cache in <15ms)

### User satisfaction metrics

Technical metrics (Hit Rate, latency) don't measure business outcomes. Track user-facing success:

**Answer helpfulness**: Did the user find the answer useful?
- **Target**: >4.0/5.0 average rating (thumbs up/down or 1-5 stars)
- **Measurement**: Log user feedback on every answer, weekly dashboard
- **Action**: If <3.5/5.0 for >20% of queries, investigate common failure patterns (review negative feedback, identify missing documents or poor retrieval)

**Task completion rate**: Did the user complete their goal without escalating to support?
- **Target**: >80% of queries result in task completion (no follow-up support ticket within 24 hours)
- **Measurement**: Track queries followed by support tickets, measure deflection rate
- **Action**: If <70% completion, RAG system isn't solving user problems (add documents, improve retrieval, enhance prompts)

**Support ticket deflection**: How many support tickets did RAG prevent?
- **Target**: 30-50% reduction in support ticket volume (for FAQ/support RAG systems)
- **Measurement**: Compare ticket volume before/after RAG launch, control for seasonality
- **Action**: If <20% deflection, RAG isn't addressing common support questions (analyze top ticket categories, add missing content)

**Query-to-answer success rate**: What fraction of queries produce an answer?
- **Target**: >95% of queries return non-empty results
- **Measurement**: Log retrieval failures (no chunks found, low similarity, timeout)
- **Action**: If <90%, investigate causes (missing content, bad embeddings, overly restrictive filters)

### Monitoring dashboard

**Weekly metrics** (track trends, alert on regressions):
- Retrieval quality: Hit Rate@5, MRR (run on labeled test set)
- Latency: p50, p95, p99 (all queries)
- User satisfaction: Average rating, task completion %, support deflection %
- System health: Error rate, timeout rate, cache hit rate

**Monthly report** (for stakeholders):
- Quality trend: Hit Rate@5 vs previous month (green if stable/improving, red if declining >5%)
- User satisfaction: Average rating, support ticket deflection (quantify value: "RAG prevented 500 support tickets = $25K savings")
- System scale: Query volume, document count, storage cost
- Roadmap: Upcoming improvements (new content, reranking upgrades, RBAC rollout)

**Alerting thresholds** (immediate action required):
- Hit Rate@5 <85% (quality regression)
- p99 latency >500ms (performance issue)
- Error rate >5% (system instability)
- User rating <3.5/5.0 for >20% of queries (user dissatisfaction)

### Quality vs cost trade-offs

Higher quality costs more. Balance based on use case:

| Use Case | Quality Target | Latency Target | Cost/1K Queries | Justification |
|----------|---------------|----------------|-----------------|---------------|
| **FAQ chatbot** | Hit Rate@5 >85% | <100ms p50 | $0.05-0.10 | High volume, cost-sensitive, moderate quality acceptable |
| **Enterprise search** | Hit Rate@5 >90% | <150ms p50 | $0.10-0.20 | Professional users, higher quality expectations, reranking justified |
| **Legal/Medical RAG** | Hit Rate@5 >95% | <200ms p50 | $0.20-0.50 | High-stakes, accuracy critical, multi-stage reranking + fine-tuned models |
| **Internal knowledge base** | Hit Rate@5 >80% | <200ms p50 | $0.02-0.05 | Low volume, employee users, basic retrieval sufficient |

**Cost optimization without sacrificing quality**:
- Use semantic caching (40-70% hit rate, 4× cost reduction)
- Skip reranking for simple queries (keyword match + short query → stage 1 sufficient)
- Use hybrid search instead of dense-only (5-10% quality gain at no extra cost)
- Batch query embeddings (10-100 queries per API call, amortize overhead)

**When to increase spend for quality**:
- User ratings <3.5/5.0 (quality too low, users frustrated)
- Support ticket deflection <20% (system not useful, wasting support team time)
- Enterprise customers require SLA (contractual quality commitment)
- High-stakes domains where errors are expensive (legal, medical, finance)

---

## Rollout & change management

Deploying RAG to production requires a phased approach to minimize risk and measure impact. This section outlines deployment strategies, A/B testing, and rollback plans.

### Phased rollout strategy

**Phase 1: Internal beta** (2-4 weeks):
- **Goal**: Validate quality and identify edge cases before customer exposure
- **Audience**: 10-20 internal users (customer support team, product team)
- **Deployment**: Separate staging environment or feature flag (0% production traffic)
- **Monitoring**:
  - Manual quality review: Sample 50-100 answers daily, label correct/incorrect
  - Retrieval metrics: Hit Rate@5, MRR on internal test set
  - Latency: p50, p99 (ensure <200ms before scaling)
  - User feedback: Daily standup with beta users, collect improvement suggestions
- **Success criteria**:
  - >80% of answers marked "correct" by internal reviewers
  - Hit Rate@5 >85% on test set
  - p99 latency <200ms
  - No critical bugs (crashes, data leakage, auth failures)
- **Duration**: 2-4 weeks (iterate on chunking, retrieval, prompts based on feedback)

**Phase 2: Limited production rollout** (4-6 weeks):
- **Goal**: Measure impact vs baseline (old system) with real users
- **Audience**: 10-25% of users (A/B test split by user ID)
- **Deployment**: Feature flag with gradual ramp (10% → 15% → 25% weekly)
- **Monitoring**:
  - A/B test metrics: Compare RAG group vs baseline (old keyword search or no-search baseline)
  - User satisfaction: Average rating (RAG vs baseline), task completion % (RAG vs baseline)
  - Performance: Latency (ensure RAG <100ms vs baseline <50ms acceptable if quality higher)
  - Business impact: Support ticket deflection, time-to-resolution
- **Success criteria**:
  - User satisfaction: RAG rating >3.8/5.0 AND >10% higher than baseline
  - Task completion: RAG >75% AND >5% higher than baseline
  - No latency regression: RAG p95 <200ms (acceptable vs baseline p95 <100ms if quality justifies)
- **Duration**: 4-6 weeks (allow statistical significance, N>1000 queries per group)

**Phase 3: Full rollout** (2-4 weeks):
- **Goal**: Scale to 100% of users while monitoring for issues
- **Audience**: All users (100% traffic)
- **Deployment**: Gradual ramp (25% → 50% → 75% → 100% over 2-4 weeks)
- **Monitoring**:
  - Same metrics as phase 2, but watch for scale issues (latency degradation, cost spikes, error rate)
  - Infrastructure: Vector DB CPU/memory, query queue depth, storage growth
  - Cost: Actual spend vs forecast (embedding API, vector DB, reranking)
- **Success criteria**:
  - User satisfaction remains stable (>3.8/5.0) as traffic scales
  - Latency remains within SLA (p95 <200ms at 100% traffic)
  - Error rate <2% (no infrastructure bottlenecks)
- **Duration**: 2-4 weeks (validate stability at full scale)

**Phase 4: Deprecate baseline** (30 days):
- **Goal**: Remove old system after proving RAG stability
- **Action**: Keep old system running in shadow mode (not serving traffic, but available for emergency rollback)
- **Timeline**: 30 days after 100% rollout (allows time to catch edge cases)
- **Final step**: After 30 days of stable production, decommission old system (shut down servers, archive code)

### A/B testing framework

**What to test**:
- RAG vs baseline (old keyword search or no search)
- Retrieval strategies (dense-only vs hybrid vs two-stage reranking)
- Chunking strategies (500-token vs 1000-token chunks)
- Reranking models (6-layer vs 12-layer cross-encoder)

**How to split traffic**:
- **User-based**: Split by user ID hash (consistent experience per user, recommended)
- **Query-based**: Split by query ID hash (faster results, but user sees mixed experience)

**Sample size calculation**:
- **Minimum**: 1000 queries per group (statistical power for 5% effect size)
- **Recommended**: 5000+ queries per group (detect 2-3% effect size)
- **Timeline**: 2-4 weeks at 100 queries/day per group

**Metrics to compare**:
- User satisfaction: Average rating (A vs B), statistical significance (t-test, p<0.05)
- Task completion: % of queries leading to completed task (A vs B)
- Latency: p95 latency (A vs B), ensure no regression
- Cost: Cost per 1K queries (A vs B), ROI calculation

**Example A/B test result**:

| Metric | Baseline (A) | RAG (B) | Improvement | Significant? |
|--------|-------------|---------|-------------|--------------|
| Average rating | 3.2/5.0 | 4.1/5.0 | +28% | ✓ (p<0.001) |
| Task completion | 65% | 82% | +17% | ✓ (p<0.01) |
| p95 latency | 80ms | 120ms | +50% slower | - (acceptable) |
| Cost per 1K queries | $0.02 | $0.08 | 4× higher | - (justified by quality) |

**Decision**: Deploy RAG (B). User satisfaction and task completion significantly improved, cost increase justified by support deflection (82% vs 65% completion = 17% fewer support tickets).

### Rollback plan

**When to roll back** (automatic or manual triggers):
- Hit Rate@5 drops <75% (critical quality regression)
- User satisfaction <3.0/5.0 for >30% of queries (user revolt)
- p99 latency >1000ms for >5% of queries (system overload)
- Error rate >10% (system instability)

**Rollback mechanism**:
- **Feature flag**: Instant switchback to baseline (no code deploy, toggle flag from 100% → 0%)
- **Blue-green deployment**: Route traffic from RAG cluster (blue) to baseline cluster (green) via load balancer
- **Timeline**: <5 minutes for emergency rollback (manual), <1 minute for automated rollback (if alerting triggers)

**Post-rollback actions**:
1. **Incident report**: Root cause analysis within 24 hours (why did quality degrade?)
2. **Fix**: Address root cause (bad chunk size, broken reranker, vector DB issue)
3. **Validation**: Test fix on staging with internal beta users (2-4 days)
4. **Retry rollout**: Restart phase 2 (10% traffic) after fix validated

**Preventing rollback scenarios**:
- Run staging environment with production-scale data (catch issues before production)
- Automated regression tests: Weekly eval on test set, alert if Hit Rate@5 drops >5%
- Gradual rollout: 10% → 25% → 50% → 100% over weeks (catch issues at small scale)
- Shadow mode: Run new system in parallel with baseline, compare results before switching traffic

### Change management

**Communication plan**:
- **Internal stakeholders**: Weekly status updates (PM, engineering leads, support team)
- **External customers**: Release notes for major changes (new content indexed, improved search)
- **Support team**: Training on RAG system (how it works, when to escalate, known limitations)

**Training materials**:
- **User documentation**: "How to search effectively" (query tips, advanced filters)
- **Support playbook**: "Troubleshooting RAG issues" (user reports bad answer → check logs, identify root cause)
- **Engineering runbook**: "RAG system operations" (restart vector DB, reindex documents, tune efSearch)

**Stakeholder expectations**:
- **Users**: "Search results are now semantic (understands intent) but may differ from old keyword search"
- **Support team**: "30-50% fewer tickets expected (FAQ deflection), but new types of issues (RAG hallucinations)"
- **Engineering**: "Ongoing maintenance 2-4 hours/week (index updates, monitoring, query optimization)"

---

## Operational risks & mitigation

RAG systems have failure modes that impact business outcomes. This section identifies risks and provides mitigation strategies for production operations.

### Infrastructure risks

**Risk 1: Vector database failure**
- **Impact**: RAG system down, users see error messages, support tickets spike
- **Probability**: Medium (1-2 incidents per year for self-hosted, <1 per year for managed services)
- **Mitigation**:
  - **High availability**: Run 3-node Qdrant cluster with automatic failover (survives 1 node failure)
  - **Backup/restore**: Daily backups to S3, tested restore process (<1 hour RTO)
  - **Monitoring**: Alert on vector DB downtime (Prometheus + PagerDuty), <5 min detection
  - **Fallback**: Graceful degradation to keyword search (if old system still available) or error message with support contact
- **Cost**: +50% infrastructure cost (3 nodes instead of 1), +2 hours/month for backup testing

**Risk 2: Embedding API outage** (OpenAI, Cohere, etc.)
- **Impact**: Cannot embed new queries, retrieval fails, users see errors
- **Probability**: Low (99.9% SLA for OpenAI API, ~8 hours downtime per year)
- **Mitigation**:
  - **Retry logic**: Exponential backoff (retry after 1s, 2s, 4s, 8s), fail after 5 attempts
  - **Fallback**: Use cached query embeddings (if query seen before), or fallback to keyword search (BM25)
  - **Multi-provider**: Use secondary embedding provider (Cohere) if OpenAI fails (requires maintaining embeddings from both models)
  - **Self-hosted backup**: Deploy local embedding model (Sentence Transformers on GPU) as fallback (latency 50-100ms vs 20ms for API)
- **Cost**: +$100-200/month for GPU server (if self-hosted backup), +1 week engineering time (multi-provider setup)

**Risk 3: Index corruption** (vector DB data loss)
- **Impact**: Chunks missing, retrieval returns wrong results, user satisfaction drops
- **Probability**: Low (<1% per year for managed services, 1-2% for self-hosted)
- **Mitigation**:
  - **Daily backups**: Snapshot vector DB to S3 daily (Qdrant, Pinecone support snapshots)
  - **Validation checks**: Daily automated job counts chunks, compares to expected count, alerts if <95%
  - **Tested restore**: Quarterly disaster recovery drill (restore from backup, validate retrieval quality <1 hour)
  - **Immutable source**: Keep original documents in S3/database, can rebuild index from source if corruption detected
- **Cost**: +10% storage cost (backups), +4 hours/quarter (disaster recovery testing)

### Quality risks

**Risk 4: Retrieval quality degradation over time**
- **Impact**: Hit Rate@5 drops from 90% → 75%, users report "search doesn't work anymore"
- **Probability**: High (60% of RAG systems experience quality drift within 6 months)
- **Root causes**:
  - Document drift: New documents have different writing style/vocabulary than indexed documents
  - Embedding model mismatch: Fine-tuned embeddings become stale as domain terminology evolves
  - Metadata staleness: Filters based on outdated categories (e.g., "2024 product docs" when it's 2026)
- **Mitigation**:
  - **Weekly evaluation**: Run automated eval on test set (200-500 labeled queries), alert if Hit Rate@5 <85%
  - **Quarterly retraining**: Fine-tune embeddings on recent queries (every 3-6 months)
  - **Document freshness monitoring**: Track average document age, alert if >80% chunks >1 year old
  - **User feedback loop**: Tag low-rated answers (rating <3/5) for manual review, identify systemic issues
- **Cost**: 2-4 hours/week (monitoring), 1-2 days/quarter (embedding fine-tuning)

**Risk 5: Embedding model change breaks index**
- **Impact**: Must reindex all documents (1-3 days downtime or dual-index migration)
- **Probability**: Low (1-2 times per year, when upgrading to better model)
- **Mitigation**:
  - **Staged migration**: Build new index with new embeddings in parallel, A/B test, hot-swap after validation (no downtime)
  - **Backwards compatibility**: If possible, quantize old embeddings to match new model (PCA, not always viable)
  - **Plan ahead**: Budget 1 week engineering time + 1-3 days reindexing time for model upgrades
- **Cost**: $10-50 (reindexing API costs), 1 week engineering time (staged migration)

### Security risks

**Risk 6: Data leakage (multi-tenant RAG)**
- **Impact**: Customer A sees Customer B's data, contract breach, lawsuit, customer churn (100% of affected customers)
- **Probability**: Medium (if RBAC not implemented), Low (if RBAC implemented correctly)
- **Mitigation**:
  - **Implement RBAC**: Hard filtering at query time, physically block unauthorized chunks (see [RBAC section]({{ site.baseurl }}/docs/genai/rag/retrieval/#hard-security-filtering-rbac))
  - **Penetration testing**: Quarterly security audit (attempt to access other tenants' data)
  - **Audit logging**: Log all queries with user_id, alert on cross-tenant access attempts
  - **Regression testing**: Automated test suite validates RBAC filters work correctly (run on every deployment)
- **Cost**: $50K-150K (initial RBAC implementation), 1 day/quarter (security audits)

**Risk 7: PII/PHI exposure in logs**
- **Impact**: GDPR/HIPAA violation, regulatory fine ($100K-$20M), audit failure
- **Probability**: Medium (if logging queries without sanitization)
- **Mitigation**:
  - **Sanitize logs**: Redact SSNs, credit cards, emails from query logs before storing (regex-based or NER model)
  - **Encryption**: Encrypt logs at rest (S3 server-side encryption) and in transit (TLS)
  - **Access control**: Limit log access to authorized engineers (IAM roles), audit log access
  - **Retention policy**: Delete logs after 90 days (or regulatory minimum, e.g., 7 years for HIPAA)
- **Cost**: 1-2 days engineering time (log sanitization), minimal ongoing cost

### Cost risks

**Risk 8: Unexpected cost spike**
- **Impact**: Monthly bill 5-10× higher than forecast, budget overrun
- **Probability**: Medium (30% of teams experience 2× cost vs estimate in first 6 months)
- **Root causes**:
  - Query volume higher than forecast (virality, marketing campaign)
  - Embedding API rate-limited (retry loops generate 10× API calls)
  - Reindexing too frequently (daily instead of weekly)
- **Mitigation**:
  - **Cost alerting**: Set budget alerts (AWS Cost Anomaly Detection, DataDog), alert if daily spend >2× forecast
  - **Rate limiting**: Cap API calls per minute (prevent retry storms)
  - **Query quotas**: Limit queries per user (10-100 per day), prevent abuse
  - **Incremental indexing**: Only reindex changed documents, not full corpus
- **Cost**: 1 day engineering time (alerting setup), minimal ongoing cost

### Business continuity

**Disaster recovery targets**:
- **RTO (Recovery Time Objective)**: <1 hour (vector DB restore from backup)
- **RPO (Recovery Point Objective)**: <24 hours (daily backups, lose at most 1 day of new documents)

**Backup strategy**:
- **What to backup**: Vector DB snapshots, original documents (S3), metadata (database), config (git)
- **Frequency**: Daily automated backups (vector DB), continuous backup (S3, database)
- **Storage**: Multi-region S3 (survive regional outage), 30-day retention

**Incident response**:
1. **Detection**: Automated alerting (vector DB down, quality <80%, latency >500ms)
2. **Triage**: On-call engineer investigates (check dashboards, logs, error rates)
3. **Mitigation**: Rollback to baseline (if feature flag available) or restart services (if transient issue)
4. **Recovery**: Restore from backup (if data corruption) or redeploy (if config issue)
5. **Post-mortem**: Root cause analysis, action items to prevent recurrence

**On-call rotation**:
- **Coverage**: 24/7 for production systems (or business hours for internal tools)
- **Team size**: 3-5 engineers (rotate weekly, prevent burnout)
- **Expected load**: 1-2 incidents per quarter (most are false alarms or minor issues)

---

## Best practices

**Use two-stage retrieval.** Stage 1 (two-tower or hybrid) retrieves 50-100 candidates in <50ms. Stage 2 (cross-encoder) reranks to 5-10 in <50ms. Total latency <100ms, MRR improvement 10-20%.

**Default to hybrid search.** Combine dense and sparse retrieval with alpha=0.5. Adjust alpha based on your queries — favor dense (alpha=0.7) for conversational queries, favor sparse (alpha=0.3) for keyword/entity search.

**Optimize stage 1 for recall, stage 2 for precision.** Stage 1 must have high recall (>90%) because missed chunks won't be reranked. Use efSearch=50-100 for HNSW. Stage 2 fixes precision — it's okay if stage 1 ranks the correct chunk 20th as long as it's in the top 50.

**Cache query embeddings for common queries.** If 20% of queries are repeated ("What is RAG?", "How do I..."), cache their embeddings. Reduces stage 1 latency from 30ms to <1ms.

**Monitor retrieval metrics independently of generation metrics.** Track Hit Rate@5, MRR, and retrieval latency. If the LLM answers incorrectly but the correct chunk was in the top 5, retrieval worked — fix the prompt or model. If the correct chunk isn't in the top 5, fix retrieval (better embeddings, hybrid search, reranking).

**Use metadata filtering for explicit constraints only.** Filter by year, category, or author when users specify them. Don't filter by relevance score — let reranking handle relevance. Overly restrictive filters miss relevant chunks.

**Tune efSearch based on latency budget.** If you have <50ms for stage 1, set efSearch=20 (90% recall). If you have 100ms, set efSearch=100 (98% recall). The reranker will compensate for lower recall if you're reranking 100+ candidates.

**Rerank at least 2× more candidates than you return.** If the LLM needs 5 chunks, retrieve 100 in stage 1, rerank to 50, return top 5. This gives the reranker enough candidates to correct stage 1 errors.

**Profile latency at each stage.** Measure stage 1 (retrieval), stage 2 (reranking), and total latency separately. If stage 1 is slow (>50ms), tune efSearch or switch algorithms (IVF, PQ). If stage 2 is slow (>100ms), reduce candidates or use a smaller reranker (6-layer vs 12-layer).

**Use query decomposition for multi-hop questions.** "Compare X and Y" or "How does A affect B?" often need separate retrievals. Decompose with an LLM, retrieve for each sub-query, deduplicate, rerank with the original query.

**A/B test retrieval strategies.** Baseline (dense-only) vs hybrid vs two-stage reranking. Measure Hit Rate@5 and MRR on 200+ labeled queries. If two-stage improves MRR by <5%, the added complexity isn't justified.

**Use MMR for multi-faceted queries, skip for precision search.** If queries span multiple topics ("vacation planning", "compare solutions"), apply MMR with lambda=0.5 to ensure diverse results. For specific factual lookups ("What is X?"), skip MMR — redundancy confirms correctness.

**Apply compression after reranking, not before.** Rerank first (100 → 50 candidates), then compress (50 → 20 with compression). Compression is expensive (LLM calls or embedding computations) — only apply to high-quality reranked candidates. Compressing before reranking wastes resources on low-quality chunks.

**Choose parent-child strategy by document structure.** Linear documents (articles, blogs) → sentence window with window_size=3-5. Hierarchical documents (technical docs, research papers) → auto-merging with threshold=0.6. Match retrieval strategy to content structure for optimal context expansion.

**Monitor token usage vs quality for parent-child retrieval.** Sentence window and auto-merging increase token usage 5-10×. Track LLM context usage and answer quality separately. If token usage is high but quality isn't improving, reduce window_size or increase merge_threshold to be more selective about expansion.

**Use three-stage retrieval for latency-accuracy balance.** Two-tower (1M → 100 candidates in 30-50ms) → ColBERT (100 → 50 in 50-100ms) → cross-encoder (50 → 5 in 50-100ms). Total 130-250ms fits within 300ms SLA for interactive RAG. ColBERT bridges the gap between two-tower speed and cross-encoder accuracy.

**Implement semantic caching for FAQ/support systems.** 40-70% hit rate, 4-8× faster (100ms → 12-25ms), 40-73% cost savings. Use 0.90-0.95 similarity threshold for balanced precision/recall. Set domain-appropriate TTL (1 hour dynamic, 24 hours stable, 7 days reference). Add event-driven invalidation for critical accuracy.
