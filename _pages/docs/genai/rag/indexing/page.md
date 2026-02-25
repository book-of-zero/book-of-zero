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
  - [Why preprocessing matters](#why-preprocessing-matters)
  - [Cleaning PDF documents](#cleaning-pdf-documents)
  - [Cleaning web scrapes](#cleaning-web-scrapes)
  - [Cleaning OCR output](#cleaning-ocr-output)
  - [Preprocessing checklist](#preprocessing-checklist)
  - [Trade-offs](#trade-offs)
- [Chunking strategies](#chunking-strategies)
  - [Character splitting](#character-splitting)
  - [Recursive character splitting](#recursive-character-splitting)
  - [Document-specific splitting](#document-specific-splitting)
  - [Semantic chunking](#semantic-chunking)
  - [Agentic splitting](#agentic-splitting)
- [Embedding models](#embedding-models)
- [Metadata strategy](#metadata-strategy)
  - [What metadata to track](#what-metadata-to-track)
  - [Metadata structure](#metadata-structure)
  - [Metadata in practice](#metadata-in-practice)
- [Vector storage](#vector-storage)
- [Chunking evaluation](#chunking-evaluation)
- [Workflow: building an index](#workflow-building-an-index)
  - [Prototype pipeline](#prototype-pipeline)
  - [Production pipeline](#production-pipeline)
  - [Incremental updates](#incremental-updates)
  - [Multi-document types](#multi-document-types)
- [Common pitfalls](#common-pitfalls)
- [Cost planning & budgeting](#cost-planning--budgeting)
  - [Indexing costs](#indexing-costs)
  - [Storage costs](#storage-costs)
  - [Query costs](#query-costs)
  - [Total cost of ownership (TCO)](#total-cost-of-ownership-tco)
- [Build vs buy decision framework](#build-vs-buy-decision-framework)
  - [Comparison matrix](#comparison-matrix)
  - [Decision tree](#decision-tree)
  - [Vendor-specific recommendations](#vendor-specific-recommendations)
  - [Migration paths](#migration-paths)
  - [Cost breakeven analysis](#cost-breakeven-analysis)
- [Team resource planning](#team-resource-planning)
  - [Team size by project phase](#team-size-by-project-phase)
  - [Ongoing maintenance](#ongoing-maintenance)
  - [Skills assessment matrix](#skills-assessment-matrix)
  - [Timeline benchmarks](#timeline-benchmarks)
  - [Staffing recommendations by scale](#staffing-recommendations-by-scale)
- [Best practices](#best-practices)

---

## Key concepts

**Chunk**: The atomic unit of retrieval in RAG. A contiguous text segment that contains enough context to be meaningful in isolation. Too small and you lose context; too large and you dilute relevance.

**Chunk size**: The target length in tokens or characters. Typical range: 500-1000 tokens. Smaller chunks increase precision but reduce context; larger chunks do the opposite.

**Chunk overlap**: The number of tokens shared between consecutive chunks. Prevents information loss at boundaries. Typical range: 10-20% of chunk size (50-200 tokens).

**Embedding**: A dense vector representation of text that encodes semantic meaning. Models map chunks to points in high-dimensional space where similar meanings cluster together.

**Vector database**: A specialized storage system optimized for approximate nearest neighbor (ANN) search over high-dimensional embeddings. Production RAG requires ANN — linear search doesn't scale past 10K chunks.

**Semantic chunking**: Splitting based on meaning rather than syntax. Uses embeddings to detect topic boundaries. 10-100× slower than recursive splitting but produces higher-quality chunks for heterogeneous documents.

---

## Data preprocessing & cleaning

Real-world documents are messy. Before chunking, you need to clean them. PDFs have headers, footers, page numbers, and watermarks. OCR produces garbled text. Web scrapes include navigation menus and ads. Skip this step and your chunks will contain garbage that ruins retrieval quality.

### Why preprocessing matters

**Problem**: You index a corporate PDF. A user queries "Q3 revenue projections" and retrieval returns chunks containing:

```
Page 12 of 87                    CONFIDENTIAL - DO NOT DISTRIBUTE
────────────────────────────────────────────────────────────────
Q3 revenue projections show 15% growth...
Copyright © 2024 Company Name. All rights reserved.
```

The embedding model wastes capacity encoding "Page 12 of 87" and "CONFIDENTIAL" instead of the actual content. Worse, if every page has "CONFIDENTIAL", the term loses discriminative power (high TF-IDF weight makes it important for retrieval, but it appears in every chunk).

**Solution**: Strip headers, footers, page numbers, watermarks, and boilerplate before chunking.

### Cleaning PDF documents

PDFs are the worst offenders. They have layout artifacts that text extraction tools blindly include.

#### Common PDF artifacts to remove

**Headers and footers**:
- Page numbers: "Page 1", "1 of 87", "12"
- Document titles: "Q3 Financial Report", "Confidential Memo"
- Watermarks: "DRAFT", "CONFIDENTIAL", "DO NOT DISTRIBUTE"
- Dates: "Printed on 2024-03-15", "Last updated: Jan 2024"

**Layout artifacts**:
- Horizontal separators: "────────────────", "================================"
- Bullets and numbering: "•", "◦", "1.", "a)", "i."
- Table borders: "|", "┌", "└", "─", "│"
- Multi-column artifacts: Text from adjacent columns mixed together

**OCR errors** (if PDF is scanned image):
- Garbled characters: "Q3 reνenue" (Greek nu instead of v), "l0% growth" (digit 0 instead of letter O)
- Missing spaces: "Q3revenue projections" (OCR missed space)
- Extra spaces: "Q 3  r e v e n u e" (OCR added spaces within words)
- Phantom characters: Random dots, tildes, or symbols from image noise

#### Cleaning pipeline

```python
import re
from typing import List
import pymupdf4llm  # Better PDF extraction than pdfplumber

def clean_pdf_text(pdf_path: str) -> str:
    """
    Extract and clean text from PDF, removing headers/footers/artifacts.

    Returns cleaned text ready for chunking.
    """
    # 1. Extract with layout awareness (preserves tables, columns)
    text = pymupdf4llm.to_markdown(pdf_path)

    # 2. Remove page numbers (various formats)
    text = re.sub(r'\b[Pp]age \d+\b', '', text)  # "Page 12"
    text = re.sub(r'\b\d+ of \d+\b', '', text)   # "12 of 87"
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Standalone numbers

    # 3. Remove common headers/footers
    text = re.sub(r'\bCONFIDENTIAL\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDO NOT DISTRIBUTE\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDRAFT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyright © \d{4}.*', '', text)  # Copyright notices

    # 4. Remove horizontal separators
    text = re.sub(r'[-─=]{3,}', '', text)  # 3+ repeated dashes/lines

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines → double newline (paragraph break)

    # 6. Remove leading/trailing whitespace
    text = text.strip()

    return text

# Usage
cleaned_text = clean_pdf_text("financial_report.pdf")
chunks = splitter.split_text(cleaned_text)  # Now chunk the cleaned text
```

#### Advanced PDF preprocessing

For production systems with heterogeneous PDFs (different layouts, fonts, languages), use ML-based document parsing:

**Document AI libraries**:
- **unstructured.io**: Detects headers/footers via layout analysis, preserves tables/images
- **docling** (IBM): ML-based document understanding, handles complex layouts
- **Azure Document Intelligence**: Cloud API, best for forms/tables/receipts
- **AWS Textract**: Similar to Azure, good for scanned documents

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, remove_punctuation

# Partition PDF with layout detection
elements = partition_pdf(
    "financial_report.pdf",
    strategy="hi_res",  # Use layout detection + OCR
    infer_table_structure=True,  # Preserve table structure
)

# Filter out headers/footers (identified by position and font size)
content_elements = [
    elem for elem in elements
    if elem.category not in ["Header", "Footer", "PageNumber"]
]

# Clean each element
cleaned_chunks = []
for elem in content_elements:
    text = elem.text
    text = clean_extra_whitespace(text)  # Normalize whitespace
    # Add more cleaning as needed
    cleaned_chunks.append(text)

# Now embed cleaned chunks
```

### Cleaning web scrapes

Web pages have navigation, ads, and boilerplate that pollute chunks.

#### Common web artifacts to remove

**Navigation elements**:
- Menus: "Home | About | Products | Contact"
- Breadcrumbs: "Home > Products > Widget > Details"
- Pagination: "Previous | 1 2 3 4 5 | Next"

**Boilerplate**:
- Cookie banners: "We use cookies to improve your experience. Accept | Decline"
- Footer text: "© 2024 Company. Terms | Privacy | Sitemap"
- Social media buttons: "Share on Facebook | Tweet | LinkedIn"

**Ads and CTAs**:
- Ad text: "Click here for 50% off!", "Limited time offer!"
- Popups: "Subscribe to our newsletter!", "Download our app!"

#### Cleaning pipeline

```python
from bs4 import BeautifulSoup
import requests

def clean_web_page(url: str) -> str:
    """
    Fetch and clean web page, extracting main content only.

    Returns cleaned text ready for chunking.
    """
    # 1. Fetch page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 2. Remove script and style tags
    for script in soup(["script", "style", "noscript"]):
        script.decompose()

    # 3. Remove navigation, footer, ads (common HTML patterns)
    for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
        element.decompose()

    # 4. Remove elements with common ad/banner class names
    for element in soup.find_all(class_=re.compile(r'ad|banner|cookie|popup|subscribe', re.I)):
        element.decompose()

    # 5. Extract main content (heuristic: look for <main> or <article> tags)
    main_content = soup.find('main') or soup.find('article') or soup.find('body')

    # 6. Get text
    text = main_content.get_text(separator='\n', strip=True)

    # 7. Remove empty lines
    lines = [line for line in text.split('\n') if line.strip()]
    text = '\n\n'.join(lines)

    return text

# Better alternative: Use specialized web scraping libraries
from trafilatura import extract

def clean_web_page_advanced(url: str) -> str:
    """Use trafilatura for better main content extraction."""
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    return text

# Usage
cleaned_text = clean_web_page_advanced("https://example.com/article")
chunks = splitter.split_text(cleaned_text)
```

### Cleaning OCR output

OCR (Optical Character Recognition) produces text from scanned images. Quality varies widely based on image resolution, font clarity, and language.

#### Common OCR errors

**Character substitution**:
- l (lowercase L) ↔ 1 (digit one) ↔ I (uppercase i)
- O (letter) ↔ 0 (zero)
- rn (two letters) ↔ m (single letter)
- cl ↔ d

**Examples**:
- "l0% growth" → should be "10% growth" (lowercase L and zero instead of one and zero)
- "reνenue" → should be "revenue" (Greek nu instead of v)
- "EXAMP1E" → should be "EXAMPLE" (digit 1 instead of letter L)

#### Cleaning pipeline

```python
import re
from spellchecker import SpellChecker

def clean_ocr_text(text: str) -> str:
    """
    Clean OCR errors (character substitutions, missing spaces).

    Returns cleaned text ready for chunking.
    """
    # 1. Fix common character substitutions
    text = re.sub(r'\bl([0O])', r'1\1', text)  # l0 → 10, lO → 1O
    text = re.sub(r'([0-9])l\b', r'\g<1>1', text)  # 0l → 01
    text = re.sub(r'\brn\b', 'm', text)  # rn → m (only whole words to avoid false positives)

    # 2. Fix missing spaces (heuristic: lowercase followed by uppercase = missing space)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # "theCompany" → "the Company"

    # 3. Fix extra spaces (OCR sometimes adds spaces within words)
    # Heuristic: single letter followed by space and another letter = likely error
    # Example: "r e v e n u e" → "revenue"
    text = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', text)  # Single letter + space + single letter

    # 4. Spell checking (expensive, use sparingly)
    # spell = SpellChecker()
    # words = text.split()
    # corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    # text = ' '.join(corrected_words)

    return text

# Advanced: Use OCR with built-in error correction
import pytesseract
from PIL import Image

def ocr_with_correction(image_path: str) -> str:
    """
    OCR with Tesseract, using language model for error correction.
    """
    # Tesseract with language model (better accuracy)
    text = pytesseract.image_to_string(
        Image.open(image_path),
        lang='eng',  # Use language model (supports multi-language)
        config='--psm 6'  # Page segmentation mode: assume uniform block of text
    )

    # Post-process
    text = clean_ocr_text(text)
    return text
```

#### When OCR quality is too low

If OCR error rate >10%, consider:
- **Re-scan at higher resolution** (300 DPI minimum, 600 DPI ideal)
- **Preprocess images**: Deskew, denoise, binarize (convert to black/white)
- **Use specialized OCR models**: Azure Computer Vision, AWS Textract, Google Cloud Vision (cloud APIs trained on billions of images)
- **Human-in-the-loop**: Flag low-confidence OCR for manual review (Tesseract provides confidence scores per word)

### Preprocessing checklist

Before chunking, validate your preprocessing:

**Quality checks**:
- Sample 10-20 documents, manually review cleaned text
- Check for remaining artifacts: page numbers, headers, garbled OCR
- Verify important content not removed: tables, equations, technical terms

**Automated validation**:
- Length check: Cleaned text should be 70-95% of original length (too little = over-cleaning, too much = under-cleaning)
- Word distribution: Check most common words (should be domain terms, not "Page", "Confidential", "Copyright")
- Embedding quality: Embed sample chunks, manually verify similar chunks cluster together (artifacts would dilute similarity)

**Monitoring in production**:
- Track cleaning metrics: % of documents with artifacts removed, average text length before/after
- Alert on anomalies: If >20% of documents have <50% text remaining, cleaning may be too aggressive

### Trade-offs

**Over-cleaning vs under-cleaning**:
- **Over-clean**: Remove too much (e.g., remove all numbers, remove all punctuation) → lose important content (dates, dollar amounts, company names)
- **Under-clean**: Remove too little → embeddings waste capacity on boilerplate, retrieval returns chunks with artifacts

**Heuristic-based vs ML-based**:
- **Heuristic** (regex, keyword matching): Fast, deterministic, but brittle (breaks on edge cases)
- **ML-based** (unstructured.io, docling): Slower, more robust, but requires GPU and handles edge cases better

**When to invest in preprocessing**:
- **Low priority**: Clean, well-formatted documents (markdown, well-structured PDFs)
- **Medium priority**: Mixed document types (some clean, some messy) → basic heuristics sufficient
- **High priority**: Messy documents (scanned PDFs, web scrapes, OCR) → invest in ML-based parsing and human review

---

## Chunking strategies

Splitting text into chunks is the foundation of RAG quality. Start with recursive character splitting for most use cases. Move to document-specific or semantic approaches when you've measured that your chunking strategy is the bottleneck.

### Character splitting

Split text every N characters. This is the baseline you use to prove that smarter chunking matters.

```python
def character_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text at fixed character intervals.

    Why this fails: Splits mid-sentence, mid-word, mid-thought.
    When to use: Never in production. Baseline for ablation studies only.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap prevents boundary information loss

    return chunks
```

**Why this fails**: You get chunks like `"...the capital of France is Par"` and `"ris. The Eiffel Tower..."`. The retrieval system can't match `"capital of France"` to the first chunk because `"Paris"` is split. The LLM can't answer from the second chunk because it lacks the question context.

**When it succeeds anyway**: Short, uniform documents where every N characters happens to align with natural boundaries (rare). Or when your chunk size is so large (5000+ tokens) that boundary problems are statistically rare (but then you lose retrieval precision).

### Recursive character splitting

Split on paragraph breaks, then sentences, then words, respecting a maximum chunk size. This is your default chunking strategy.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# separators prioritized: first try \n\n (paragraphs), then \n (lines), then spaces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Target size in characters (not tokens)
    chunk_overlap=200,      # 20% overlap prevents boundary information loss
    separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraph → sentence → word boundaries
    length_function=len,    # Use token count in production (e.g., tiktoken)
)

chunks = splitter.split_text(document_text)
```

**How it works**: The splitter tries to split at paragraph boundaries (`\n\n`) first. If a paragraph exceeds `chunk_size`, it tries sentence boundaries (`. `). If a sentence is too long, it splits at word boundaries (` `). Only if a single word exceeds the limit does it fall back to character splitting.

**Why this works**: Chunks respect natural boundaries 95% of the time. You get coherent units like a full paragraph or a complete explanation. The overlap ensures that context spanning boundaries (e.g., "As mentioned above, the solution...") is captured.

**When to use**: Start here. Use recursive splitting unless you have document-specific structure to exploit or you've measured that semantic chunking is worth the cost.

**Parameter tuning**:
- **chunk_size**: 500 tokens for high precision (FAQ, product specs), 1000 tokens for general use, 1500+ tokens for long-context models (GPT-4, Claude) where you want maximum context per retrieval
- **chunk_overlap**: 10-20% of chunk_size — below 10% you lose boundary information, above 20% you're storing redundant data without significant quality gains
- **length_function**: Use `len` for prototypes, switch to `tiktoken.encoding_for_model("gpt-4").encode` in production to count tokens accurately

### Document-specific splitting

Use document structure (markdown headers, code syntax, PDF layout) to guide splitting. When your documents have explicit hierarchy, exploit it.

#### Markdown documents

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Split at header boundaries, preserving hierarchy as metadata
headers_to_split_on = [
    ("#", "h1"),      # Top-level sections
    ("##", "h2"),     # Subsections
    ("###", "h3"),    # Sub-subsections
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # Keep headers in chunk text for context
)

md_chunks = markdown_splitter.split_text(markdown_document)

# Each chunk has metadata: {"h1": "Introduction", "h2": "Architecture", "h3": "Components"}
# Use metadata for filtering: "only search the 'API Reference' section"
```

**Why this works**: Headers signal topic boundaries. A chunk containing everything under "## Authentication" is semantically coherent — it won't mix login logic with database configuration. The metadata enables filtered search: "find authentication methods in the API section" restricts retrieval to chunks with `h1="API Reference"` and `h2="Authentication"`.

**When to use**: Technical documentation, wikis, READMEs, books. Anywhere the author used headers to organize information.

#### Code files

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Code-aware separators respect syntax boundaries
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",  # Also supports: js, java, cpp, go, rust, etc.
    chunk_size=1000,
    chunk_overlap=200,
)

# Prioritizes splitting at: class definitions → function definitions → blank lines → statements
code_chunks = code_splitter.split_text(python_code)
```

**Why this works**: Splitting mid-function produces unreadable chunks. Code-aware splitting tries to keep functions intact, splits at class boundaries, and preserves docstrings with their implementations.

**When to use**: Code search, codebase Q&A, automated documentation generation.

#### PDFs and multi-modal documents

PDFs are not text files. They have layout (columns, tables, sidebars), formatting (bold, italics), and embedded non-text elements (images, charts).

```python
from unstructured.partition.pdf import partition_pdf

# Extract structured elements with layout awareness
elements = partition_pdf(
    "document.pdf",
    strategy="hi_res",  # OCR + layout detection (slow but accurate)
    extract_images_in_pdf=True,  # Extract images for multi-modal embedding
)

# elements is a list of typed objects: Title, NarrativeText, Table, Image
# Chunk by grouping related elements
chunks = []
current_chunk = []

for element in elements:
    if element.category == "Title":
        # Start new chunk at section boundaries
        if current_chunk:
            chunks.append("\n".join([e.text for e in current_chunk]))
        current_chunk = [element]
    else:
        current_chunk.append(element)
```

**Why this works**: You don't split mid-table or separate a chart from its caption. Each chunk is a semantically complete unit (a section with its title, a table with its header).

**When to use**: Scientific papers, reports, presentations. Documents where layout conveys structure.

### Semantic chunking

Split when the topic changes, not at fixed intervals. Use embeddings to detect semantic boundaries.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import SemanticChunker

embeddings = OpenAIEmbeddings()

# Split when cosine similarity between consecutive sentences drops below threshold
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # Split at bottom 25% of similarities
    breakpoint_threshold_amount=25,
)

semantic_chunks = semantic_splitter.split_text(document_text)
```

**How it works**:
1. Split text into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentence embeddings
4. Where similarity drops (topic boundary), start a new chunk
5. Group sentences between boundaries into chunks

**Why this works**: Topic changes correlate with embedding distance. A document discussing "training neural networks" then "deploying models" will show a similarity drop at the transition. Semantic chunking detects that boundary even if there's no paragraph break.

**When to use**: Heterogeneous documents where structure doesn't match semantics (transcripts, web scrapes, unformatted reports). Or when you've measured that recursive splitting produces chunks that mix topics and hurt retrieval quality.

**Cost**: 10-100× slower than recursive splitting. You're embedding every sentence. For a 10,000-word document (100 sentences), that's 100 embedding API calls. At $0.0001/1K tokens, this costs $0.10 per document vs $0.001 for one-shot embedding after chunking.

**Ablation test**: Run recursive splitting and semantic chunking on 100 documents. Evaluate retrieval quality (see [Chunking evaluation](#chunking-evaluation)). If semantic chunking improves Hit Rate@5 by <5%, the cost isn't justified.

### Agentic splitting

Use an LLM to decide how to chunk. Experimental and expensive, but handles edge cases that rule-based systems miss.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

# LLM decides chunk boundaries and writes summaries
prompt = PromptTemplate.from_template("""
You are a document chunking expert. Split this text into coherent chunks.

Rules:
- Each chunk should be 300-800 tokens
- Split at topic boundaries
- Each chunk must be self-contained (understandable without prior context)

Document:
{document}

Output JSON: [{{"chunk_text": "...", "summary": "..."}}]
""")

response = llm.invoke(prompt.format(document=document_text))
agentic_chunks = parse_json(response.content)  # Extract chunks from LLM response
```

**Why this might work**: The LLM understands nuance. It can merge a short paragraph with the next section if they're topically related, or split a long paragraph if it contains two distinct ideas.

**Why this fails**: Expensive ($0.01-0.10 per document), slow (5-10 seconds), and non-deterministic. Hard to debug when chunking quality regresses. LLMs are also prone to violating constraints (outputting 2000-token chunks when you asked for 800).

**When to use**: High-stakes applications where chunking quality is critical and cost/latency don't matter (legal document analysis, medical literature review). Or as a research baseline to establish the quality ceiling.

**Practical alternative**: Use agentic chunking to label 100 documents, then train a lightweight model to predict boundaries. Deploy the model instead of the LLM.

---

## Embedding models

Your chunks are text. Your vector database expects numbers. Embedding models bridge the gap.

### Model selection

| Model | Dimensions | Speed (ms/chunk) | Quality (MTEB) | Cost ($/1M tokens) | When to use |
|-------|-----------|------------------|----------------|-------------------|-------------|
| **text-embedding-3-small** (OpenAI) | 1536 | 50 | 62.3 | $0.02 | Default for prototypes. Fast, cheap, good enough. |
| **text-embedding-3-large** (OpenAI) | 3072 | 80 | 64.6 | $0.13 | Better quality when Hit Rate@5 is too low with small model. |
| **bge-large-en-v1.5** (open) | 1024 | 30 | 63.9 | Free (self-hosted) | Cost-sensitive production. Requires GPU for fast inference. |
| **instructor-xl** (open) | 768 | 100 | 61.8 | Free (self-hosted) | Domain-specific embeddings with instruction tuning. |
| **multilingual-e5-large** | 1024 | 40 | 61.5 | Free (self-hosted) | Multi-language retrieval (100+ languages). |

**MTEB**: Massive Text Embedding Benchmark. Average score across 56 retrieval tasks. Higher is better. Difference of 2 points (e.g., 62 → 64) typically translates to 5-10% improvement in Hit Rate@5.

### Fine-tuning embeddings

Pre-trained models work out of the box. Fine-tuning helps when your domain differs from the training data (medical jargon, legal language, internal company terminology).

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('bge-large-en-v1.5')

# Create training pairs: (query, relevant_chunk, irrelevant_chunk)
train_examples = [
    InputExample(texts=["what is HIPAA?", "HIPAA is the Health Insurance Portability...", "The capital of France..."]),
    # ... 1000+ examples
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Fine-tune with triplet loss: pull queries closer to relevant chunks, push away from irrelevant
train_loss = losses.TripletLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

**When to fine-tune**: You've tried 3+ pre-trained models and retrieval quality is still below target. You have 1000+ labeled query-document pairs. The cost of collecting labels and training is justified by improved quality.

**When to skip**: Pre-trained models already hit your quality target. You don't have labeled data. Your domain is covered by the model's training data (general web text, Wikipedia).

### Trade-offs

- **Dimensions**: Higher dimensions capture more nuance but require more storage and slower search. 768 dimensions = 3KB per chunk (float32). For 1M chunks, that's 3GB of vectors. 3072 dimensions = 12GB.
- **Speed**: Embedding is on the indexing critical path. At 50ms/chunk, indexing 10K documents (100K chunks) takes 1.4 hours. At 30ms/chunk, it's 50 minutes. Batch embedding (10-100 chunks per API call) amortizes overhead.
- **Quality**: Better embeddings improve retrieval, but retrieval is one part of RAG. If your LLM is bad at using context, better embeddings won't help. Fix the prompt or the model first.

---

## Metadata strategy

Metadata transforms retrieval from "find similar text" to "find the right text from the right source at the right time." Well-structured metadata enables filtering, ranking, attribution, and context enrichment.

### What metadata to track

**Document-level metadata** (applies to all chunks from a document):

- **source**: Document origin (filename, URL, database ID). Enables tracing chunks back to source.
- **doc_type**: Document category (pdf, markdown, code, email, ticket). Enables type-specific filtering.
- **created_at / updated_at**: Timestamps (ISO 8601 format). Enables recency filtering and time-based ranking.
- **author / owner**: Creator or responsible party. Enables authority-based ranking and access control.
- **category / tags**: Topic classification (engineering, legal, sales). Enables domain filtering.
- **language**: Document language (en, es, fr). Enables multi-language retrieval.
- **version**: Document version number. Enables version-specific retrieval and deduplication.
- **access_level**: Permission level (public, internal, confidential). Enables basic access control and soft boosting by user role.
- **roles**: List of roles authorized to access (["engineering", "compliance"]). Enables hard security filtering via RBAC (Role-Based Access Control).
- **departments**: Department or organizational unit (["legal", "finance"]). Enables hierarchical permission filtering.
- **permissions**: Granular permission tags (["read", "write", "admin"]). Enables fine-grained access control.
- **owner_id**: User or team ID that owns the document. Enables ownership-based filtering and attribution.

**Chunk-level metadata** (specific to individual chunks):

- **chunk_index**: Position within document (0, 1, 2...). Enables ordering and context reconstruction.
- **section / heading**: Section hierarchy from document structure (h1, h2, h3). Enables section filtering.
- **page_number**: Page in original document (for PDFs). Enables page-specific retrieval and citation.
- **char_count / token_count**: Chunk size metrics. Enables quality monitoring and deduplication.

**Derived metadata** (computed during indexing):

- **embedding_model**: Model used to generate embedding (all-mpnet-base-v2). Enables compatibility checking.
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
    "author": "hr-team",
    "created_at": "2024-01-15T10:30:00Z",
    "page_number": 5,
    "section": "Benefits",
}

# Bad: Nested metadata (slower to query, harder to index)
metadata = {
    "document": {
        "source": "handbook.pdf",
        "type": "pdf",
    },
    "content": {
        "category": "hr",
        "section": "Benefits",
    },
}
```

### Metadata in practice

Attach metadata during indexing:

```python
# Document with rich metadata
chunks = splitter.split_text(document_text)
embeddings = model.encode(chunks)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    metadata = {
        # Document-level
        "source": "hipaa_guide.pdf",
        "doc_type": "pdf",
        "category": "compliance",
        "author": "legal-team",
        "created_at": "2024-01-15T10:30:00Z",
        "version": "v2.3",
        "access_level": "internal",

        # RBAC (Role-Based Access Control)
        "roles": ["legal", "compliance", "engineering"],  # Who can access
        "departments": ["legal"],  # Org-level filtering
        "permissions": ["read"],  # Granular permissions
        "owner_id": "legal-team",  # Owner attribution

        # Chunk-level
        "chunk_index": i,
        "section": extract_section(chunk),  # From document structure
        "page_number": extract_page(chunk),
        "token_count": len(chunk.split()),

        # Derived
        "embedding_model": "all-mpnet-base-v2",
        "indexed_at": datetime.utcnow().isoformat(),
        "doc_hash": hashlib.sha256(document_text.encode()).hexdigest(),
    }

    # Store chunk with metadata
    client.upsert(
        collection_name="documents",
        points=[PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk, **metadata})]
    )
```

**Metadata enables**:
- **Filtered retrieval**: "Only search PDF documents from legal-team"
- **Recency boosting**: Rank recent documents higher
- **Attribution**: Show users "Source: hipaa_guide.pdf, Page 5"
- **Access control**: Filter by user permissions (soft boosting via access_level)
- **Hard security filtering**: Physically block unauthorized chunks via RBAC metadata (roles, departments, permissions). See [retrieval filtering]({{ site.baseurl }}/docs/genai/rag/retrieval/#hard-security-filtering-rbac) for implementation.
- **Deduplication**: Skip reindexing unchanged documents (via doc_hash)

### Contextual chunk headers (CCH)

Chunks extracted from documents often lack context about where they came from. The sentence "Nike has committed to reducing emissions by 50%" is meaningless without knowing it's from the "Nike Climate Impact Report 2025" under "Environmental Commitments." Contextual Chunk Headers (CCH) prepend document and section context to chunks before embedding.

**Impact**: Adding context headers can increase similarity scores from 0.1 to 0.92 for relevant queries. Chunks become self-contained and retrieval precision improves dramatically.

**How it works**:

1. Parse document structure (title, section headers, subsections)
2. For each chunk, prepend a context header with document title and section hierarchy
3. Embed the contextualized chunk (header + original text)
4. Store both the full contextualized text and original chunk text
5. At retrieval time, search uses contextualized embeddings but LLM receives clean chunks (without redundant headers)

**Implementation**:

```python
def extract_document_structure(document: str, doc_title: str):
    """
    Parse document into sections with hierarchy.

    Returns: List of (section_header, section_text) tuples
    """
    # Simple markdown parser (assumes ## headers)
    sections = []
    current_section = None
    current_text = []

    for line in document.split('\n'):
        if line.startswith('##'):
            # Save previous section
            if current_section:
                sections.append((current_section, '\n'.join(current_text)))

            # Start new section
            current_section = line.strip('# ')
            current_text = []
        else:
            current_text.append(line)

    # Save last section
    if current_section:
        sections.append((current_section, '\n'.join(current_text)))

    return sections

def create_contextual_chunks(document: str, doc_title: str, chunk_size: int = 500):
    """
    Create chunks with contextual headers.

    Each chunk is prepended with:
    - Document title
    - Section hierarchy
    - Optional: Subsection, document metadata

    Returns: List of (contextualized_chunk, original_chunk, metadata) tuples
    """
    sections = extract_document_structure(document, doc_title)
    all_chunks = []

    for section_header, section_text in sections:
        # Chunk the section text
        chunks = split_text(section_text, chunk_size=chunk_size, overlap=50)

        for i, chunk in enumerate(chunks):
            # Create context header
            context_header = f"""Document: {doc_title}
Section: {section_header}

"""
            # Contextualized version (for embedding)
            contextualized_chunk = context_header + chunk

            # Store both versions
            all_chunks.append({
                'contextualized': contextualized_chunk,  # Embed this
                'original': chunk,  # Return this to LLM
                'metadata': {
                    'doc_title': doc_title,
                    'section': section_header,
                    'chunk_index': i,
                }
            })

    return all_chunks

# Usage during indexing
doc_title = "Nike Climate Impact Report 2025"
document_text = load_document("nike_climate_report.md")

chunks = create_contextual_chunks(document_text, doc_title, chunk_size=500)

# Embed contextualized versions
contextualized_texts = [chunk['contextualized'] for chunk in chunks]
embeddings = model.encode(contextualized_texts)

# Store both versions
for chunk, embedding in zip(chunks, embeddings):
    client.upsert(
        collection_name="documents",
        points=[PointStruct(
            vector=embedding.tolist(),
            payload={
                'contextualized_text': chunk['contextualized'],  # For reference
                'original_text': chunk['original'],  # Return this to LLM
                'doc_title': chunk['metadata']['doc_title'],
                'section': chunk['metadata']['section'],
                'chunk_index': chunk['metadata']['chunk_index'],
            }
        )]
    )

# At retrieval time:
# - Search uses contextualized embeddings (finds "Nike Climate Impact Report > Environmental Commitments")
# - LLM receives original_text (clean chunk without redundant headers)
query_results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)

# Extract original chunks for LLM
llm_context = [result.payload['original_text'] for result in query_results]
```

**Context header formats**:

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

**Hierarchical** (full section path):
```
Document: Nike Climate Impact Report 2025
Path: Report > Environmental Strategy > Commitments > Emissions

Nike has committed to reducing emissions by 50% by 2030...
```

**When to use CCH**:
- Documents with clear hierarchical structure (reports, academic papers, technical docs, legal documents)
- Chunks that lack self-contained context ("Nike has committed..." — committed to what?)
- Multi-document corpora where queries need to distinguish sources
- Precision is critical and you can afford extra token cost

**When to skip CCH**:
- Documents are already self-contained (FAQs, product descriptions, tweets)
- Single-document retrieval where context is obvious
- Token costs are prohibitive (CCH adds 10-50 tokens per chunk)
- Simple chunking already achieves acceptable retrieval quality

**Trade-offs**:
- **Pro**: Dramatically improves retrieval precision (0.1 → 0.92 similarity in studies)
- **Pro**: Chunks become self-contained and portable
- **Pro**: Reduces out-of-context responses
- **Con**: Increases token count per chunk (10-50 tokens for header)
- **Con**: Higher embedding costs (more tokens to embed)
- **Con**: Requires preprocessing to extract document structure
- **Con**: Storage overhead (store both contextualized and original versions)

**Optimization**: Use abbreviated headers for frequently repeated metadata:

```python
# Standard (50 tokens)
"Document: Nike Climate Impact Report 2025\nSection: Environmental Commitments\n\n"

# Abbreviated (20 tokens)
"Doc: Nike Climate 2025 | Env Commitments\n\n"

# Balance precision with token cost
```

**Production considerations**:
- Parse document structure during ingestion (one-time cost)
- Cache section headers to avoid recomputing for every chunk
- A/B test: measure retrieval quality with and without CCH on your corpus
- Monitor embedding costs (CCH increases embedding API calls by 10-30%)
- Consider CCH only for high-value documents where precision matters most

---

## Vector storage

You've chunked documents and embedded them. Now you need a database optimized for "find the 5 closest vectors to this query vector" across millions of chunks.

### In-memory vs persistent

**In-memory** (FAISS, Annoy): Fast (µs latency), no network overhead, but limited by RAM. Works for <10M chunks (~30GB for 1536 dimensions). Requires reloading on restart.

```python
import faiss
import numpy as np

# Create HNSW index (Hierarchical Navigable Small World graph)
dimension = 1536
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter, number of connections per node

# Add embeddings
embeddings = np.array([...])  # Shape: (num_chunks, dimension)
index.add(embeddings)

# Search for 5 nearest neighbors
query_embedding = np.array([...])  # Shape: (dimension,)
distances, indices = index.search(query_embedding[np.newaxis, :], k=5)
```

**Persistent** (Pinecone, Weaviate, Qdrant, Milvus): Store vectors on disk, distributed across machines. Scales to billions of chunks. Network latency (5-20ms) but handles larger-than-RAM datasets.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Insert chunks
points = [
    PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk, "metadata": {...}})
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
]
client.upsert(collection_name="documents", points=points)

# Search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding.tolist(),
    limit=5,
)
```

**When to use in-memory**: <1M chunks, single-machine deployment, you need <10ms retrieval latency, you can tolerate reindexing on restarts.

**When to use persistent**: >1M chunks, multi-machine scaling, shared index across services, you need filtering (metadata queries), you can tolerate 20-50ms retrieval latency.

### Index types

| Algorithm | Recall@10 | Latency | Memory | Build time | When to use |
|-----------|-----------|---------|--------|------------|-------------|
| **HNSW** (Hierarchical Navigable Small World) | 95-99% | <10ms | High (3-5× vectors) | Fast (minutes) | Default. Best recall/latency trade-off. |
| **IVF** (Inverted File Index) | 80-95% | <5ms | Low (1.5× vectors) | Fast (minutes) | Memory-constrained. Acceptable recall loss. |
| **PQ** (Product Quantization) | 70-90% | <2ms | Very low (0.1× vectors) | Slow (hours) | Billion-scale. Need compression. |
| **Flat** (brute-force) | 100% | Linear in N | 1× vectors | Instant | <10K chunks. Ground truth for ablations. |

**HNSW** is your default. It's what Pinecone, Weaviate, and Qdrant use under the hood.

#### HNSW tuning

```python
# efConstruction: candidate list size during index build (higher = better recall, slower build)
# M: number of bi-directional links per node (higher = better recall, more memory)
index = faiss.IndexHNSWFlat(dimension, M=32)
index.hnsw.efConstruction = 200

# efSearch: candidate list size during search (higher = better recall, slower search)
index.hnsw.efSearch = 50

# Rule of thumb:
# - M=32, efConstruction=200, efSearch=50: balanced (95% recall, <10ms)
# - M=64, efConstruction=400, efSearch=100: high recall (98%, <20ms)
# - M=16, efConstruction=100, efSearch=20: fast (90% recall, <5ms)
```

**Recall@10**: Probability that the true top-10 nearest neighbors are in the approximate top-10. 95% recall means 9.5 out of 10 results are correct. The missing result is usually rank 11-15, not a distant outlier.

**When recall matters**: High-stakes retrieval where missing the best chunk degrades quality (medical Q&A, legal search). Optimize for recall.

**When recall doesn't matter**: You're retrieving 50 chunks and reranking to 5 (see [RAG Retrieval]({{ site.baseurl }}/docs/genai/rag/retrieval/page)). The reranker fixes approximate errors. Optimize for latency.

### Production patterns

1. **Build index offline**: Don't block user queries while indexing new documents. Use a staging index, build it fully, then hot-swap.
2. **Shard by metadata**: Split index by time (recent vs archive), domain (product docs vs blog posts), or access pattern. Search shards in parallel or selectively (only search recent docs for time-sensitive queries).
3. **Monitor recall**: Track end-to-end metrics (did the LLM answer correctly?) but also track retrieval metrics (was the correct chunk in the top-5?). If retrieval recall drops, your index parameters are wrong.
4. **Quantize in production**: Use float16 or int8 embeddings to halve memory/storage. Quality loss is <1% for most models.

```python
# Quantize embeddings to float16
embeddings_fp16 = embeddings.astype(np.float16)
index.add(embeddings_fp16)

# Or use 8-bit quantization with FAISS
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist=100, m=8, nbits=8)
```

---

## Chunking evaluation

Chunking is a proxy task. The real goal is retrieval quality (did we find the right chunk?) and generation quality (did the LLM answer correctly?). But you can measure chunking quality independently.

### Intrinsic metrics (chunk quality)

Measures chunk properties without running retrieval or generation.

**Chunk coherence**: Do chunks contain semantically related sentences? Embed each sentence in a chunk, compute pairwise cosine similarity. High similarity = coherent. Low similarity = mixed topics.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_coherence(chunk: str) -> float:
    """Higher is better. Coherent chunks have intra-chunk similarity > 0.7."""
    sentences = chunk.split('. ')
    if len(sentences) < 2:
        return 1.0

    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)

    # Average pairwise similarity (excluding diagonal)
    n = len(similarities)
    return (similarities.sum() - n) / (n * (n - 1))

# Good: 0.8 (sentences are about the same topic)
# Bad: 0.4 (chunk mixes unrelated topics)
```

**Boundary preservation**: Do chunks respect natural boundaries (paragraphs, sections)? Count splits that occur mid-sentence.

```python
def mid_sentence_splits(chunks: list[str]) -> float:
    """Lower is better. 0 = all splits respect sentence boundaries."""
    violations = 0
    for chunk in chunks:
        # If chunk doesn't end with sentence-ending punctuation, it's a mid-sentence split
        if chunk.strip() and chunk.strip()[-1] not in '.!?':
            violations += 1
    return violations / len(chunks)
```

**Chunk size variance**: Are chunks uniformly sized? High variance means some chunks are too small (insufficient context) or too large (diluted relevance).

```python
import numpy as np

def chunk_size_variance(chunks: list[str]) -> float:
    """Lower is better. Coefficient of variation (std/mean)."""
    lengths = [len(chunk.split()) for chunk in chunks]
    return np.std(lengths) / np.mean(lengths)

# Good: <0.3 (uniform sizes)
# Bad: >0.5 (wildly varying sizes)
```

### Extrinsic metrics (retrieval quality)

Measures whether chunks enable successful retrieval. Requires labeled query-document pairs.

**Hit Rate@k**: Fraction of queries where the correct chunk appears in the top-k results.

```python
def hit_rate_at_k(queries: list[str], ground_truth_chunks: list[str], retrieved_chunks: list[list[str]], k: int) -> float:
    """Queries, ground truth, and retrieved results. Returns fraction where ground truth is in top-k."""
    hits = 0
    for query, true_chunk, retrieved in zip(queries, ground_truth_chunks, retrieved_chunks):
        if true_chunk in retrieved[:k]:
            hits += 1
    return hits / len(queries)

# Good: >0.9 (top-5 contains the right chunk 90% of the time)
# Bad: <0.7 (missing relevant chunks)
```

**MRR (Mean Reciprocal Rank)**: Average of 1/rank where rank is the position of the first correct chunk.

```python
def mean_reciprocal_rank(queries: list[str], ground_truth_chunks: list[str], retrieved_chunks: list[list[str]]) -> float:
    """Higher is better. 1.0 = correct chunk always rank 1. 0.5 = average rank 2."""
    reciprocal_ranks = []
    for true_chunk, retrieved in zip(ground_truth_chunks, retrieved_chunks):
        try:
            rank = retrieved.index(true_chunk) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # Not found
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**NDCG@k**: Normalized Discounted Cumulative Gain. Like MRR but accounts for graded relevance (some chunks are more relevant than others). See [ML Evaluation]({{ site.baseurl }}/docs/machine-learning/evaluation/page/#workflow-ranking) for details.

### Evaluation workflow

1. **Create test set**: 100-500 queries with labeled correct chunks. Sample from real user queries if available, otherwise write synthetic queries covering major use cases.
2. **Baseline**: Character splitting or recursive splitting with default parameters.
3. **Ablations**: Try different chunk sizes (500, 1000, 1500 tokens), overlaps (0%, 10%, 20%), and splitting strategies (recursive, markdown, semantic).
4. **Measure**: Compute Hit Rate@5, MRR, chunk coherence for each configuration.
5. **Select**: Choose configuration that maximizes Hit Rate@5 subject to latency constraints. If two configurations have similar Hit Rate (difference <2%), pick the simpler one (recursive over semantic).

**Example results**:

| Configuration | Hit Rate@5 | MRR | Chunk Coherence | Indexing Time (1K docs) |
|---------------|------------|-----|-----------------|-------------------------|
| Character split (baseline) | 0.65 | 0.42 | 0.55 | 5 min |
| Recursive (500 tokens, 10% overlap) | 0.82 | 0.61 | 0.74 | 8 min |
| Recursive (1000 tokens, 20% overlap) | 0.85 | 0.65 | 0.78 | 10 min |
| Semantic | 0.88 | 0.68 | 0.82 | 4 hours |

**Interpretation**: Recursive (1000 tokens, 20% overlap) is the winner. Semantic chunking improves Hit Rate by 3% but costs 24× more time. Not worth it unless you're in the high-stakes regime.

---

## Workflow: building an index

Indexing is not a single function call. It's a pipeline: load documents → chunk → embed → store → validate. This section walks through end-to-end indexing workflows from prototype to production.

### Prototype pipeline

Start simple. Get documents indexed and searchable in <50 lines of code.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load documents
documents = [
    "RAG stands for Retrieval-Augmented Generation...",
    "Paris is the capital of France...",
    # ... load from files, database, API
]

# Chunk with recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = splitter.split_text("\n\n".join(documents))  # Join with separator to preserve doc boundaries

# Embed chunks
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Start with exact search (Flat) for prototypes
index.add(embeddings.astype('float32'))

# Save index and chunks
faiss.write_index(index, "index.faiss")
np.save("chunks.npy", np.array(chunks, dtype=object))

print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
```

**What this gives you**: A working RAG index in minutes. Exact search (IndexFlatL2) is fine for <10K chunks. No optimization, no metadata, no error handling — but it works.

**When to use**: Prototypes, demos, proof-of-concepts. Validating that RAG helps before investing in infrastructure.

### Production pipeline

Production indexing adds: document metadata tracking, incremental updates, ANN indexes, error handling, and monitoring.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionIndexer:
    """Production-ready document indexer with metadata, error handling, and monitoring."""

    def __init__(self, collection_name: str = "documents"):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name

        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {collection_name}")

    def index_document(self, doc_id: str, text: str, metadata: Dict) -> int:
        """
        Index a single document with metadata.

        Returns: Number of chunks created
        """
        try:
            # Generate document hash for change detection
            doc_hash = hashlib.sha256(text.encode()).hexdigest()

            # Check if document already indexed (idempotency)
            existing = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
                limit=1,
            )

            if existing[0] and existing[0][0].payload.get("doc_hash") == doc_hash:
                logger.info(f"Document {doc_id} unchanged, skipping")
                return 0

            # Delete old chunks if document exists
            if existing[0]:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
                )
                logger.info(f"Deleted old chunks for document {doc_id}")

            # Chunk document
            chunks = self.splitter.split_text(text)

            # Embed chunks
            embeddings = self.model.encode(chunks, show_progress_bar=False)

            # Prepare points with metadata
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = hashlib.sha256(f"{doc_id}-{i}".encode()).hexdigest()[:16]  # Deterministic IDs
                points.append(
                    PointStruct(
                        id=int(point_id, 16) % (2**63),  # Convert to int for Qdrant
                        vector=embedding.tolist(),
                        payload={
                            "doc_id": doc_id,
                            "doc_hash": doc_hash,
                            "chunk_index": i,
                            "text": chunk,
                            "indexed_at": datetime.utcnow().isoformat(),
                            **metadata,  # User-provided metadata (category, author, timestamp, etc.)
                        },
                    )
                )

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Indexed document {doc_id}: {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            raise

    def index_batch(self, documents: List[Dict]) -> Dict[str, int]:
        """
        Index multiple documents.

        documents: List of dicts with keys: doc_id, text, metadata
        Returns: Dict with stats
        """
        stats = {"total_docs": len(documents), "total_chunks": 0, "failed": 0}

        for doc in documents:
            try:
                chunks = self.index_document(
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    metadata=doc.get("metadata", {}),
                )
                stats["total_chunks"] += chunks
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"Document {doc.get('doc_id')} failed: {e}")

        logger.info(f"Batch complete: {stats}")
        return stats

# Usage
indexer = ProductionIndexer(collection_name="knowledge_base")

documents = [
    {
        "doc_id": "doc_001",
        "text": "RAG stands for Retrieval-Augmented Generation...",
        "metadata": {"category": "AI", "author": "team", "year": 2024},
    },
    # ... more documents
]

stats = indexer.index_batch(documents)
print(f"Indexed {stats['total_chunks']} chunks from {stats['total_docs']} documents")
```

**What this adds**:
- **Idempotency**: Reindexing the same document doesn't create duplicates
- **Change detection**: Document hashing skips unchanged documents
- **Metadata**: Track document source, category, timestamp for filtered search
- **Error handling**: Failed documents don't crash the pipeline
- **Monitoring**: Logging at every step for debugging and auditing

**When to use**: Production systems serving real users. Multi-document collections that update over time.

### Incremental updates

Don't rebuild the entire index when you add 10 new documents. Incremental indexing appends new chunks without touching existing ones.

```python
# Add new documents without full rebuild
new_documents = [
    {"doc_id": "doc_101", "text": "New content...", "metadata": {"category": "updates"}},
    {"doc_id": "doc_102", "text": "More new content...", "metadata": {"category": "updates"}},
]

# Index new documents (existing documents unchanged)
stats = indexer.index_batch(new_documents)

# Update a single document (old chunks deleted, new chunks added)
updated_doc = {
    "doc_id": "doc_001",  # Existing document
    "text": "Updated RAG content...",  # New text
    "metadata": {"category": "AI", "author": "team", "year": 2024, "updated": True},
}
indexer.index_document(**updated_doc)
```

**Why this matters**: Rebuilding a 1M chunk index takes hours. Incremental updates take seconds. For document collections that change daily (news, support tickets, internal docs), incremental indexing is required.

**When to rebuild**:
- Chunk size or overlap parameters change (chunks incompatible with old chunks)
- Embedding model changes (embeddings incompatible with old embeddings)
- Index algorithm changes (HNSW → IVF, M parameter tuning)

### Multi-document types

Real systems index multiple document types: PDFs, markdown, code, web pages. Use document-specific splitters and add type metadata.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from unstructured.partition.pdf import partition_pdf

class MultiTypeIndexer(ProductionIndexer):
    """Indexer that handles multiple document types."""

    def __init__(self, collection_name: str = "documents"):
        super().__init__(collection_name)

        # Initialize type-specific splitters
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200,
        )

    def chunk_by_type(self, text: str, doc_type: str) -> List[str]:
        """Route to appropriate splitter based on document type."""

        if doc_type == "markdown":
            md_chunks = self.markdown_splitter.split_text(text)
            # Markdown splitter returns objects with metadata, extract text
            return [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    for chunk in md_chunks]

        elif doc_type == "code":
            return self.code_splitter.split_text(text)

        elif doc_type == "pdf":
            # PDF requires special handling (layout-aware extraction)
            elements = partition_pdf(text)  # text is file path for PDFs
            # Group elements into chunks
            chunks = []
            current = []
            for elem in elements:
                if elem.category == "Title" and current:
                    chunks.append("\n".join([e.text for e in current]))
                    current = [elem]
                else:
                    current.append(elem)
            if current:
                chunks.append("\n".join([e.text for e in current]))
            return chunks

        else:  # Default: plain text
            return self.text_splitter.split_text(text)

    def index_document(self, doc_id: str, text: str, metadata: Dict) -> int:
        """Override to use type-specific chunking."""
        doc_type = metadata.get("type", "text")

        # Use type-specific chunker
        chunks = self.chunk_by_type(text, doc_type)

        # Embed and store (rest is same as parent class)
        embeddings = self.model.encode(chunks, show_progress_bar=False)

        # ... (rest of indexing logic)
        # For brevity, assume similar logic to ProductionIndexer

        return len(chunks)

# Usage
indexer = MultiTypeIndexer(collection_name="multi_type_kb")

documents = [
    {"doc_id": "readme", "text": "# API Docs\n## Authentication...", "metadata": {"type": "markdown"}},
    {"doc_id": "utils.py", "text": "def compute():\n    ...", "metadata": {"type": "code", "language": "python"}},
    {"doc_id": "report.pdf", "text": "/path/to/report.pdf", "metadata": {"type": "pdf"}},
]

indexer.index_batch(documents)
```

**Why type-aware chunking matters**: A PDF split mid-table is useless. A code file split mid-function is unreadable. Markdown without headers loses context. Type-specific splitters respect document structure.

**Metadata for type filtering**: Tag chunks with `type="code"` or `type="pdf"`. At retrieval time, filter by type: "only search markdown documentation" or "only search Python code".

---

## Common pitfalls

**Chunking without overlap**: Chunks at document boundaries lose context. If chunk N ends with "The solution is" and chunk N+1 starts with "to use caching", neither chunk can answer "What's the solution?". Add 10-20% overlap.

**Using character count instead of token count**: A 1000-character chunk might be 250 tokens or 400 tokens depending on the text. Models have token limits, not character limits. Use `tiktoken` to count tokens accurately.

**Ignoring chunk size distribution**: If 10% of chunks are <100 tokens, they lack context. If 10% are >2000 tokens, they're too broad. Monitor the distribution. Outliers hurt retrieval quality.

**Reindexing everything on every change**: Incremental updates are 100× faster than full rebuilds. Only rebuild when parameters change (chunk size, embedding model, index algorithm).

**Embedding before chunking**: Don't embed full documents then try to split them. Chunk first, embed chunks. Embeddings represent fixed-length vectors — you can't split a vector without re-embedding.

**Losing document metadata**: If you only store chunk text, you can't filter by document attributes (author, date, category). Always attach document metadata to chunks.

**Mixing incompatible embeddings**: If you reindex with a new embedding model (all-mpnet-base-v2 → text-embedding-3-large), you must rebuild the entire index. New embeddings and old embeddings live in different spaces — they can't be compared.

**Skipping evaluation**: You don't know if recursive splitting is better than semantic chunking until you measure Hit Rate@5 on real queries. Intuition fails. Evaluate every strategy change.

**Chunking code like prose**: Code has syntax. A chunk containing `def foo(x:` without the closing `)` is syntactically invalid and confuses retrieval. Use code-aware splitters.

**Trusting layout-agnostic PDF extraction**: `pdfplumber.extract_text()` returns a string with all layout information lost. Tables become unreadable. Use layout-aware parsers (`unstructured`, `pymupdf4llm`) that preserve structure.

**Hardcoding chunk size**: Different use cases need different chunk sizes. FAQ queries want small chunks (500 tokens) for precision. Long-form explanations want large chunks (1500 tokens) for context. Tune chunk size per use case.

**Not monitoring index growth**: Indexes grow unbounded if you never delete old chunks. A document updated 100 times creates 100× the chunks if you don't delete old versions. Monitor index size and implement TTL or version limits.

---

## Cost planning & budgeting

RAG indexing costs money — embedding API calls, storage, compute time. Business stakeholders need concrete dollar amounts to approve budgets and evaluate ROI. This section breaks down costs for typical deployment scales.

### Indexing costs

Indexing incurs one-time costs for generating embeddings and initial index creation.

**Embedding API costs** (OpenAI text-embedding-3-small, as of February 2026):
- **Standard tier**: $0.02 per 1M tokens
- **Batch tier** (asynchronous, 24-hour window): $0.01 per 1M tokens (50% discount)

**Example calculations**:

| Document Count | Avg Tokens/Doc | Total Tokens | Standard Cost | Batch Cost |
|----------------|----------------|--------------|---------------|------------|
| 10,000 docs | 500 | 5M tokens | $0.10 | $0.05 |
| 100,000 docs | 500 | 50M tokens | $1.00 | $0.50 |
| 1,000,000 docs | 500 | 500M tokens | $10.00 | $5.00 |
| 1,000,000 docs | 2000 (long docs) | 2B tokens | $40.00 | $20.00 |

**Compute time** (for embedding generation):
- Local embedding models (Sentence Transformers): Free (self-hosted), ~2-4 hours for 1M chunks on 4-core CPU with GPU
- API-based (OpenAI): Embedding time included in API cost, ~30-60 minutes for 1M chunks (rate-limited by API)

**Cost optimization strategies**:
- Use Batch API (50% savings) for initial indexing when 24-hour latency is acceptable
- Use self-hosted embedding models ([bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) to eliminate API costs (requires GPU: ~$100-200/month for dedicated server)
- Batch documents before embedding (10-100 docs per API call) to reduce overhead and improve throughput

### Storage costs

Vector storage is an ongoing monthly cost that scales with document count and vector dimensions.

**Storage size calculation**:
```
Storage (GB) = num_chunks × (vector_dim × 4 bytes + metadata_size)

Example (1M chunks, 768-dim vectors, 500 bytes metadata):
= 1M × (768 × 4 + 500) bytes
= 1M × 3,572 bytes
= 3.57 GB
```

**Vector database pricing** (as of February 2026):

| Provider | Pricing Model | Cost for 1M chunks (768-dim) | Notes |
|----------|--------------|------------------------------|-------|
| **Pinecone** | $0.33/GB/month storage + read/write units | ~$70-200/month | Serverless, fully managed, minimum $50/month |
| **Weaviate Cloud** | $25 per 1M vector dimensions/month | ~$20-50/month (Shared Cloud) | Open-source core, multi-tenancy support |
| **Qdrant Cloud** | Pay-as-you-go, $0.014/hour for clusters | ~$100-300/month | Free 1GB tier, scales with cluster size |
| **Qdrant Self-Hosted** | Infrastructure only (AWS/GCP) | ~$20-100/month (server) | Open-source, requires DevOps, full control |
| **FAISS In-Memory** | Infrastructure only | ~$50-200/month (RAM-heavy server) | Not persistent, <10M chunks, no managed service |

**Production storage recommendations**:
- **<100K chunks**: FAISS in-memory (cheapest for small scale)
- **100K-1M chunks**: Qdrant Cloud or Weaviate Cloud (balance cost/features)
- **>1M chunks**: Qdrant self-hosted (best cost at scale) or Pinecone (best managed experience)

**Storage cost optimization**:
- Quantize embeddings to float16 (50% storage savings, <1% quality loss)
- Use Product Quantization (PQ) for 4-8× compression (5-10% quality loss, acceptable for large indexes)
- Implement document TTL (time-to-live) to automatically delete stale chunks (e.g., delete chunks >1 year old)
- Monitor metadata size — verbose metadata (long text fields) increases storage cost disproportionately

### Query costs

Retrieval costs depend on query volume and reranking complexity.

**Per-query cost breakdown**:

| Operation | Cost (per 1K queries) | Notes |
|-----------|----------------------|-------|
| **Query embedding** (OpenAI API) | $0.02 (standard) or $0.01 (batch) | Same as indexing cost, most queries use standard tier |
| **Vector search** (managed DB) | Included in storage cost | Pinecone charges via read units (~$0.01-0.05 per 1K queries) |
| **Vector search** (self-hosted) | Compute cost only | ~$0.001-0.01 per 1K queries (amortized server cost) |
| **Reranking** (cross-encoder, GPU) | ~$0.10 per 1K queries | If reranking 50 candidates per query on GPU |

**Monthly query cost example** (10K queries/day = 300K queries/month):

| Configuration | Monthly Cost |
|---------------|--------------|
| **Basic**: OpenAI embeddings + Pinecone + no reranking | $6 (embedding) + $70 (storage/reads) = **~$76** |
| **Advanced**: OpenAI embeddings + Qdrant Cloud + cross-encoder reranking | $6 (embedding) + $100 (storage) + $30 (reranking) = **~$136** |
| **Self-hosted**: Local embeddings + Qdrant self-hosted + cross-encoder | $0 (embedding) + $50 (server) = **~$50** |

**Cost optimization strategies**:
- Implement semantic caching (cache results for similar queries, 40-70% cache hit rate reduces query costs by 40-70%)
- Use self-hosted embedding models to eliminate per-query API costs (one-time GPU investment: ~$100-200/month)
- Skip reranking for <50 candidates (two-tower retrieval often sufficient for top-10 results)
- Batch queries when possible (internal analytics, batch jobs) to use Batch API pricing

### Total cost of ownership (TCO)

**Example TCO for 1M documents, 10K queries/day**:

| Cost Component | Managed (Pinecone) | Hybrid (Qdrant Cloud) | Self-Hosted (Qdrant) |
|----------------|-------------------|----------------------|---------------------|
| **Initial indexing** | $10 (API) | $10 (API) | $0 (local embeddings) |
| **Monthly storage** | $150 | $100 | $50 (server) |
| **Monthly queries** | $6 (embedding) + included | $6 (embedding) + included | $0 (local) |
| **Reranking** | Optional: +$30 | Optional: +$30 | Optional: +$30 |
| **Ops overhead** | 0 FTE (fully managed) | 0.1 FTE (mostly managed) | 0.5 FTE (self-managed) |
| **Total monthly** | **~$156-186** | **~$106-136** | **~$50-80 + 0.5 FTE** |
| **Annual TCO** | **~$1,872-2,232** | **~$1,272-1,632** | **~$600-960 + ops labor** |

**Decision factors**:
- **Managed (Pinecone)**: Best for <3 engineers, fast time-to-market, budget >$150/month
- **Hybrid (Qdrant Cloud)**: Best for 3-10 engineers, balance cost/features, budget $100-150/month
- **Self-hosted**: Best for >10 engineers, DevOps expertise, budget <$100/month, high query volume (cost advantage at scale)

**When cost justifies investment**: RAG systems that reduce support costs (deflect 30-50% of tickets), enable new revenue (enterprise contracts requiring search), or prevent compliance violations (RBAC = avoid HIPAA/GDPR fines) typically achieve ROI within 6-12 months. See [retrieval RBAC business value]({{ site.baseurl }}/docs/genai/rag/retrieval/#hard-security-filtering-rbac) for ROI calculations.

**Sources**:
- [OpenAI Embeddings Pricing (Feb 2026)](https://platform.openai.com/docs/pricing)
- [Pinecone Pricing Guide](https://www.pinecone.io/pricing/)
- [Qdrant Cloud Pricing](https://qdrant.tech/pricing/)
- [Weaviate Cloud Pricing](https://weaviate.io/pricing)
- [Vector Database Cost Comparison (2026)](https://rahulkolekar.com/vector-db-pricing-comparison-pinecone-weaviate-2026/)

---

## Build vs buy decision framework

Choosing between self-hosted (Qdrant, Milvus, FAISS) and managed services (Pinecone, Weaviate Cloud) is a business decision, not just a technical one. The right choice depends on team size, budget, scale, and ops expertise.

### Comparison matrix

| Factor | Self-Hosted (Qdrant/Milvus) | Managed SaaS (Pinecone/Weaviate Cloud) |
|--------|------------------------------|----------------------------------------|
| **Monthly cost** (1M chunks) | $20-100 (infrastructure) | $100-300 (service fees) |
| **Upfront investment** | 2-4 weeks setup + DevOps automation | 2-4 days integration |
| **Engineering overhead** | 0.5-1 FTE for ops (monitoring, updates, backups) | ~0 FTE (fully managed) |
| **Time to production** | 8-12 weeks (setup + hardening) | 2-4 weeks (integration only) |
| **Control & customization** | Full (tune all HNSW/IVF parameters, custom plugins) | Limited (vendor-controlled configuration) |
| **Vendor lock-in** | None (open-source, migrate anytime) | High (proprietary API, migration costly) |
| **Skills required** | DevOps (K8s, monitoring, backups), vector DB tuning | API integration only (minimal ML knowledge) |
| **Scalability** | Manual (provision servers, shard indexes) | Automatic (vendor handles scaling) |
| **Support** | Community (Discord, GitHub issues) | Enterprise (Slack, dedicated support, SLA) |
| **Compliance** | DIY (SOC 2, HIPAA compliance on you) | Vendor-certified (SOC 2 Type II, HIPAA BAA available) |

### Decision tree

**Choose managed SaaS (Pinecone, Weaviate Cloud, Qdrant Cloud) if**:
- ✅ Team size <5 engineers (can't dedicate 0.5 FTE to ops)
- ✅ Time-to-market critical (need production in <4 weeks)
- ✅ Budget >$100-300/month (cost acceptable for ops savings)
- ✅ No DevOps expertise (no K8s/monitoring/backup experience)
- ✅ Enterprise compliance required (need SOC 2/HIPAA-certified vendor)
- ✅ Query volume moderate (<1M queries/day)

**Choose self-hosted (Qdrant, Milvus, FAISS) if**:
- ✅ Team size >10 engineers (have dedicated DevOps/SRE)
- ✅ Budget <$100/month (cost optimization priority)
- ✅ High query volume (>1M queries/day, SaaS costs prohibitive)
- ✅ Custom requirements (need full control over index parameters, data residency, security)
- ✅ Vendor lock-in averse (want open-source portability)
- ✅ Existing infrastructure (already run K8s clusters, monitoring, backups)

### Vendor-specific recommendations

**Pinecone**:
- **Best for**: Fast time-to-market, small teams (<5 engineers), enterprise customers requiring SOC 2/HIPAA
- **Pros**: Best UX, fully managed, automatic scaling, excellent support
- **Cons**: Highest cost ($70-500/month), vendor lock-in (proprietary API), limited customization
- **Use when**: Budget >$200/month, need production in <4 weeks, compliance critical

**Weaviate Cloud**:
- **Best for**: GraphQL users, multi-modal embeddings (text + image), open-source preference with managed option
- **Pros**: Open-source core (can self-host later), GraphQL API, multi-tenancy support, lower cost than Pinecone
- **Cons**: Smaller community than Pinecone, less polished UX, newer managed offering
- **Use when**: Need GraphQL, prefer open-source, budget $100-200/month

**Qdrant Cloud**:
- **Best for**: Cost-conscious teams with option to self-host later, need filtering (RBAC), hybrid deployment
- **Pros**: Open-source core, free 1GB tier (prototyping), flexible deployment (cloud or self-hosted), strong filtering
- **Cons**: Smaller company than Pinecone/Weaviate, newer managed offering (less mature ops)
- **Use when**: Want open-source + managed option, budget $100-200/month, need RBAC metadata filtering

**Qdrant Self-Hosted**:
- **Best for**: Cost optimization at scale, DevOps expertise, custom security/compliance requirements
- **Pros**: Lowest cost ($20-100/month), full control, no vendor lock-in, data sovereignty (on-prem or VPC)
- **Cons**: 0.5-1 FTE ops overhead, need K8s/monitoring expertise, community-only support (or paid support contract)
- **Use when**: Team >10 engineers, budget <$100/month, high query volume, on-prem/VPC required

**Milvus Self-Hosted**:
- **Best for**: Billion-scale indexes, distributed clusters, GPU acceleration, research/experimentation
- **Pros**: Best scalability (billions of vectors), GPU support, active CNCF community, free/open-source
- **Cons**: Complex setup (requires Pulsar, etcd, MinIO), steeper learning curve, 1-2 FTE ops overhead
- **Use when**: Scale >10M documents, need GPU acceleration, have SRE team for distributed systems

**FAISS In-Memory**:
- **Best for**: Prototypes, research, <1M chunks, single-machine deployment, cost-sensitive
- **Pros**: Free (library), fastest search (<1ms), full control, no network latency
- **Cons**: Not persistent (reindex on restart), no filtering (no metadata), limited to RAM size (<10M chunks)
- **Use when**: Prototyping, research, single-server app, <1M chunks, no production SLA

### Migration paths

**Start small, scale later**:
1. **Prototype** (weeks 1-2): FAISS in-memory or Pinecone free tier
2. **MVP** (weeks 3-6): Pinecone or Weaviate Cloud (fast to production)
3. **Scale** (months 6-12): Migrate to self-hosted Qdrant/Milvus when cost or control justifies ops overhead

**Graceful migration strategy**:
- Use abstraction layer (LangChain, LlamaIndex) to avoid vendor lock-in (easier to swap vector DBs)
- Maintain parallel indexes during migration (old + new system running simultaneously for 30 days)
- A/B test retrieval quality before full cutover (ensure self-hosted matches managed quality)

### Cost breakeven analysis

**When self-hosted becomes cheaper**:

| Query Volume | Managed Cost (annual) | Self-Hosted Cost (annual) | Breakeven Point |
|--------------|----------------------|--------------------------|-----------------|
| 1K queries/day | ~$1,800 (Pinecone) | ~$600 (server) + 0.5 FTE | Never (if FTE cost >$100K) |
| 10K queries/day | ~$2,200 | ~$1,200 (server + ops) | If ops <0.1 FTE or FTE <$10K |
| 100K queries/day | ~$5,000-10,000 | ~$2,000-3,000 | 6-12 months |
| 1M queries/day | ~$50,000+ | ~$10,000-15,000 | 3-6 months |

**Rule of thumb**: Self-hosted breaks even at >100K queries/day or >5M chunks when you have 0.5 FTE available for ops. Below that threshold, managed services save money (ops time > service fees).

**Sources**:
- [OpenMetal: When Self-Hosting Vector Databases Becomes Cheaper](https://openmetal.io/resources/blog/when-self-hosting-vector-databases-becomes-cheaper-than-saas/)
- [Pinecone vs Qdrant vs Weaviate Comparison (2026)](https://xenoss.io/blog/vector-database-comparison-pinecone-qdrant-weaviate)

---

## Team resource planning

RAG implementation requires specific skills and time commitments. Business stakeholders need FTE estimates and timeline projections to staff projects and plan roadmaps.

### Team size by project phase

**Prototype (2-4 weeks)**:
- **Team**: 1 engineer (backend/ML generalist)
- **Skills**: Python, basic ML knowledge (embeddings, vector similarity), API integration
- **Deliverables**: Basic indexing pipeline (chunking + embedding + storage), simple retrieval (semantic search only)
- **Quality**: MVP-level (no reranking, no RBAC, no monitoring)
- **Effort**: 40-80 hours (1 sprint)

**Production (6-12 weeks)**:
- **Team**: 2-3 engineers (1 ML engineer, 1 backend engineer, 0.5 DevOps/SRE)
- **Skills**:
  - ML engineer: Embedding model selection, chunking strategies, retrieval quality evaluation (Hit Rate@5, MRR)
  - Backend engineer: API integration, metadata management, incremental updates, error handling
  - DevOps: Vector DB deployment (if self-hosted), monitoring, backups, HA setup
- **Deliverables**: Multi-stage retrieval (hybrid search + reranking), RBAC metadata filtering, monitoring dashboards, incremental indexing, production-grade error handling
- **Quality**: Enterprise-grade (>90% Hit Rate@5, <100ms p99 latency, SOC 2-ready logging)
- **Effort**: 300-600 hours (6-12 weeks for 2-3 FTEs)

**Enterprise (2-4 months)**:
- **Team**: 3-5 engineers (2 ML, 2 backend, 1 DevOps) + 0.5 PM + 0.25 Legal/Compliance
- **Skills**:
  - ML: Advanced retrieval (ColBERT, query decomposition, HyDE), fine-tuning embeddings, custom rerankers
  - Backend: Multi-tenancy, hierarchical RBAC, audit logging, compliance reporting (SOC 2, HIPAA)
  - DevOps: Distributed vector DB clusters, auto-scaling, disaster recovery, multi-region deployment
  - PM: Requirements gathering, user acceptance testing, rollout planning
  - Legal/Compliance: GDPR/HIPAA compliance review, data processing agreements, security audits
- **Deliverables**: Multi-tenant RAG with customer isolation, hierarchical RBAC (roles + departments + ownership), comprehensive audit logs, compliance reporting (SOC 2/HIPAA evidence), A/B testing framework, semantic caching, multi-region deployment
- **Quality**: Enterprise SLA (99.9% uptime, <50ms p50 latency, full compliance)
- **Effort**: 800-1200 hours (2-4 months for 3-5 FTEs)

### Ongoing maintenance

**After launch** (steady state):
- **Team**: 0.5 FTE (rotating on-call among existing engineers)
- **Time commitment**: 2-4 hours/week (10-20% of one engineer's time)
- **Activities**:
  - **Weekly** (1-2 hours): Index updates (add/delete documents), query optimization (review slow queries), monitoring review (check dashboards for quality/latency regressions), user feedback triage (read negative ratings)
  - **Monthly** (2-4 hours): Quality evaluation (run eval suite on test set, compare vs baseline), index health check (chunk count, embedding model version, fragmentation), performance tuning (HNSW parameters, caching thresholds)
  - **Quarterly** (1-2 days): Major index rebuild (if chunk size or embedding model changes), disaster recovery test (backup/restore), capacity planning (forecast storage/query costs)
- **On-call**: 1-2 incidents per quarter (vector DB restart, slow queries, indexing failures), MTTR <1 hour for P1 (RAG system down)

**Scaling ops burden**:
- <100K chunks: 0.25 FTE (2 hours/week)
- 100K-1M chunks: 0.5 FTE (4 hours/week)
- >1M chunks: 1 FTE (full-time if self-hosted, 0.5 FTE if managed service)

### Skills assessment matrix

Rate your team's current capabilities to identify hiring/training needs:

| Skill | Required Level | Assessment Questions |
|-------|---------------|---------------------|
| **Python programming** | Intermediate | Can write production APIs (Flask/FastAPI), handle async operations, error handling? |
| **Vector databases** | Beginner (prototype)<br>Intermediate (production) | Understand ANN search (HNSW, IVF)? Can tune efSearch, M parameters? |
| **Embedding models** | Beginner | Know how to call OpenAI/Sentence Transformers APIs? Understand semantic similarity? |
| **Chunking strategies** | Beginner (prototype)<br>Intermediate (production) | Can implement recursive splitting? Understand trade-offs (size, overlap, boundaries)? |
| **Evaluation metrics** | Not required (prototype)<br>Intermediate (production) | Can compute Hit Rate@5, MRR? Set up eval pipelines with labeled test sets? |
| **DevOps (K8s, monitoring)** | Not required (managed)<br>Advanced (self-hosted) | Can deploy Qdrant on K8s? Set up Prometheus/Grafana dashboards? Manage backups? |
| **Security (RBAC, compliance)** | Intermediate (enterprise) | Understand JWT validation, metadata filtering, audit logging, HIPAA/GDPR requirements? |

**Skill gap mitigation**:
- **Hire**: ML engineer with RAG experience (6-12 months faster than training)
- **Train**: Backend engineers on vector DBs and embeddings (2-4 weeks ramp-up)
- **Consult**: Hire contractors for initial setup (4-8 weeks), transition to internal team for maintenance
- **Managed service**: Use Pinecone/Weaviate Cloud to eliminate DevOps skill requirement (reduce team size by 0.5-1 FTE)

### Timeline benchmarks

**Industry data** (based on [2026 RAG deployment studies](https://dextralabs.com/blog/enterprise-rag-llm-accuracy-blueprint-2026/)):
- **MVP using off-the-shelf tools**: 1-3 weeks (single engineer, Pinecone + LangChain)
- **Production system** (optimized retrieval, security, monitoring): 4-8 weeks (2-3 engineers)
- **Enterprise implementation** (compliance, custom infra, integrations): 2-4+ months (3-5 engineers)
- **Old model** (custom RAG pipeline per use case): 6-12 months (now obsolete with modern platforms)

**Timeline risks**:
- **Requirements churn**: Stakeholders change chunking strategy or add RBAC mid-project (+2-4 weeks per major change)
- **Data quality issues**: Source documents poorly formatted, missing metadata, need manual cleanup (+2-6 weeks)
- **Integration complexity**: Legacy systems, complex auth, multi-region deployment (+4-8 weeks)
- **Compliance delays**: Legal/security review, BAA negotiations, audit preparation (+4-12 weeks)

**Mitigation strategies**:
- Lock requirements early (write functional spec, get sign-off before implementation)
- Pilot with small dataset (10K docs) to surface data quality issues early
- Use managed services to reduce integration complexity (Pinecone + OpenAI = minimal DevOps)
- Start compliance review in parallel with development (don't wait until launch)

### Staffing recommendations by scale

| Document Scale | Query Volume | Team Size | Timeline | Annual Cost (labor + infra) |
|----------------|-------------|-----------|----------|----------------------------|
| <10K docs | <1K queries/day | 1 engineer (part-time) | 2-4 weeks | ~$20K (0.25 FTE) + $1K (Pinecone) |
| 10K-100K docs | 1K-10K queries/day | 1-2 engineers | 4-8 weeks | ~$100K (1 FTE) + $2K (managed) |
| 100K-1M docs | 10K-100K queries/day | 2-3 engineers | 8-12 weeks | ~$200K (2 FTE) + $5K (hybrid) |
| >1M docs | >100K queries/day | 3-5 engineers | 12-16 weeks | ~$400K (4 FTE) + $20K (self-hosted) |

**Cost reality check**: At typical engineer salaries ($100-150K/year), labor cost dominates infrastructure cost until >1M queries/day. Optimize for engineer productivity (use managed services) over infrastructure cost in most cases.

**Sources**:
- [RAG Implementation Timeline Benchmarks (2026)](https://www.leanware.co/insights/rag-application-development-guide)
- [Enterprise RAG Deployment Study (2026)](https://www.techment.com/blogs/rag-in-2026/)

---

## Best practices

**Start with recursive splitting.** 1000-token chunks, 200-token overlap, default separators. This works for 80% of use cases. Move to document-specific or semantic splitting only when you've measured that chunking quality is your bottleneck.

**Measure retrieval quality, not generation quality, when tuning chunking.** If Hit Rate@5 is 90% but the LLM still hallucinates, chunking is not the problem. Fix the prompt or the model.

**Overlap is not optional.** 10-20% overlap prevents information loss at boundaries. Without overlap, queries like "what happened after the merger?" fail when the merger is at the end of chunk N and the aftermath is at the start of chunk N+1.

**Use metadata for filtering.** If your documents have structure (sections, timestamps, authors), attach it to chunks. Then filter at retrieval time: "find chunks in the 'API Reference' section published after 2023". This is 10-100× faster than retrieving from the full index and filtering in post-processing.

**Don't chunk code like prose.** Use code-aware splitters that respect syntax boundaries. A chunk containing half a function is useless.

**PDF structure matters.** Don't extract raw text and split it. Use layout-aware parsers (unstructured, LlamaIndex) that preserve tables, captions, and section boundaries.

**Quantize embeddings in production.** Float16 halves storage and memory with <1% quality loss. Int8 (Product Quantization) gives 8× compression with 5-10% quality loss. Profile the trade-off on your data.

**Reindex incrementally.** Don't rebuild the entire index when you add 10 new documents. Append new chunks and rebalance (if using IVF) or insert into HNSW. Full reindex is for parameter changes (dimension, M, quantization) not content updates.

**Monitor chunk size distribution.** If 10% of chunks are <100 tokens or >2000 tokens, your splitting strategy is producing outliers. Outliers hurt retrieval quality (too small = no context, too large = diluted relevance).

**Implement hierarchical RBAC metadata for enterprise systems.** If you're building multi-tenant or compliance-driven RAG (HIPAA, GDPR), add RBAC fields at indexing time (roles, departments, owner_id). This enables hard security filtering at retrieval time (physically blocking unauthorized chunks) without reindexing. Structure permissions hierarchically (department → team → individual) for flexible query-time filtering.

**Test on real queries.** Synthetic eval queries are useful for ablations but don't capture real user behavior. Sample 500 production queries, label the correct chunks, and measure Hit Rate@5. This is your ground truth for chunking quality.
