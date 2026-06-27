# VectorDB — Vector Database Built from Scratch in C++

A working vector database with a web UI, implementing **HNSW**, **KD-Tree**, and **Brute Force** search side-by-side, plus a full **RAG pipeline** powered by a local LLM via Ollama.

> Built to demonstrate how production vector databases like Pinecone, Weaviate, and Chroma work under the hood.

## Features

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — compare speed side-by-side |
| **3 Distance Metrics** | Cosine, Euclidean, Manhattan |
| **2D PCA Visualization** | Live scatter plot showing semantic clusters form in real time |
| **Real Embeddings** | Paste any text → embedded via Ollama's `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions → HNSW retrieves relevant chunks → local LLM generates an answer |
| **REST API** | Full CRUD: insert, delete, search, benchmark, HNSW introspection |

## How It Works

```
Text → Ollama (nomic-embed-text) → HNSW Index → Semantic Search → Ollama (llama3.2) → Answer
```

**HNSW (Hierarchical Navigable Small World)** — the same algorithm used by Pinecone, Weaviate, and Milvus — builds a multilayer graph where each layer is progressively sparser. Searches start at the top layer and descend, achieving O(log N) complexity instead of brute force's O(N).

## Architecture

```cpp
BruteForce    O(N·d)      Exact, baseline
KDTree        O(log N)    Exact, axis-aligned space partitioning
HNSW          O(log N)    Approximate, multilayer graph search

VectorDB      Unified interface over all 3 (16D demo vectors)
DocumentDB    HNSW-only index for real Ollama embeddings (768D)
OllamaClient  HTTP client → /api/embeddings + /api/generate
```

**Why HNSW wins at high dimensions:** KD-Tree pruning relies on axis-aligned distance bounds, which collapse in high dimensions (curse of dimensionality) — almost no subtrees get pruned past ~20D. HNSW's graph-based approach avoids this entirely, which is why it scales to the 768D embeddings used in the RAG pipeline.

## Tech Stack

- **Backend:** C++17, [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- **Frontend:** HTML/JS (PCA scatter plot, chat UI, live benchmarking)
- **LLM/Embeddings:** Ollama (`nomic-embed-text`, `llama3.2`)

## Setup

**Prerequisites:** C++17 compiler, Git, [Ollama](https://ollama.com)

```bash
# Pull the required models
ollama pull nomic-embed-text
ollama pull llama3.2

# Clone and build
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
g++ -std=c++17 -O2 main.cpp -o db -lws2_32   # Windows
# g++ -std=c++17 -O2 main.cpp -o db          # Linux/macOS

# Run
ollama serve &      # if not already running
./db
```

Then open `http://localhost:8080`.

## Usage

- **Search** — query the 20 pre-loaded demo vectors across 4 categories (CS, Math, Food, Sports); switch between HNSW, KD-Tree, and Brute Force and compare results/speed
- **Documents** — paste any text to generate real 768D embeddings (auto-chunked into overlapping 250-word segments) and index them with HNSW
- **Ask AI** — ask natural-language questions about your uploaded documents; HNSW retrieves the most relevant chunks and llama3.2 answers using only that context

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=...&k=5&metric=cosine&algo=hnsw` | K-NN search over demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | Inspect HNSW graph structure |
| `POST` | `/doc/insert` | Embed and store a document |
| `POST` | `/doc/ask` | RAG: retrieve context + generate answer |

Full endpoint list in [main.cpp](main.cpp).

## Project Structure

```
VectorDB/
├── main.cpp        # C++ backend — HNSW, KD-Tree, Brute Force, REST API, RAG
├── httplib.h        # Single-header HTTP server (cpp-httplib)
└── index.html       # Frontend — scatter plot, chat UI, benchmarking
```

## License

MIT
