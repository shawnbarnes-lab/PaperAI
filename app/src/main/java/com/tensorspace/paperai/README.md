# PaperAI — On-Device RAG for Android

**Search your documents with AI, entirely on your phone.**

PaperAI is an Android app that uses Retrieval-Augmented Generation (RAG) to let you search personal documents with natural language. Upload PDFs, vectorize them on your GPU server, and query them offline with semantic search and an on-device LLM.

Your data never touches a cloud service. Vectorization runs on your own hardware. Search and AI answers run locally on the phone.

---

## How It Works

```
 You upload a PDF
       │
       ▼
┌──────────────────────────┐
│  Vectorizer Server (GPU) │  ← Your machine (RTX 3090/4090)
│  Flask + sentence-transformers
│  Splits into semantic chunks
│  Generates 384-dim embeddings
│  Returns .paperai file
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Android App (This Repo) │  ← Runs 100% offline after import
│  ObjectBox vector DB
│  On-device query embedding
│  Diversified similarity search
│  Gemma 3 1B LLM (on-device)
│  Shows AI summary + sources
└──────────────────────────┘
```

1. Upload PDF/TXT → server chunks it semantically and generates embeddings
2. Download `.paperai` file (JSON with chunks, embeddings, and full page text)
3. Import into app → stored in ObjectBox vector database on device
4. Search with natural language → on-device embedding → vector similarity search
5. AI generates brief summary from retrieved context → source passages displayed below

---

## Features

- **Semantic search** — finds meaning, not just keywords. "water purification" matches "making contaminated water safe to drink"
- **Source-diversified results** — large documents can't monopolize search results (max 2 per source)
- **On-device LLM** — Gemma 3 1B generates brief answers from retrieved context, no internet needed
- **Full document reconstruction** — tap any source to view the original document with matched sections highlighted
- **Extractive fallback** — if the LLM fails or isn't loaded, the app still returns the best matching sentence across all results
- **Privacy first** — documents vectorized on your hardware, search runs locally, zero telemetry

---

## Architecture

### Android App (this repo)

| File | Purpose |
|------|---------|
| `RagEngine.kt` | Orchestrates the full RAG pipeline — embedding, search, LLM, response |
| `VectorStore.kt` | ObjectBox wrapper with `searchDiversified()` — ANN search with source diversity cap |
| `LlmService.kt` | On-device Gemma 3 1B via MediaPipe LLM Inference API |
| `EmbeddingService.kt` | On-device query embedding (all-MiniLM-L6-v2 via MediaPipe) |
| `VectorImporter.kt` | Parses `.paperai` JSON into ObjectBox entities |
| `DocumentChunk.kt` | ObjectBox entity — text, 384-dim embedding, page number, source metadata |
| `DocumentPage.kt` | ObjectBox entity — full original page text for document viewer |
| `VectorizerClient.kt` | HTTP client for upload/vectorize workflow |
| `MainScreen.kt` | Jetpack Compose UI — search, AI response tab, sources tab |
| `MainViewModel.kt` | UI state management |

### Vectorizer Server ([separate repo](https://github.com/shawnbarnes-lab/paperai-vectorizer-v2))

| File | Purpose |
|------|---------|
| `server.py` | Flask API — accepts documents, returns `.paperai` files |
| `semantic_chunker.py` | Splits text into semantic chunks with accurate page tracking via character offsets |

### Models

| Model | Size | Where | Purpose |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | ~80MB | Both server + device | Text embeddings (384-dim) |
| Gemma 3 1B (int4) | ~530MB | Device only | Answer generation |

Both sides use the same embedding model so query vectors match document vectors.

### .paperai File Format (v2.1)

```json
{
  "version": 2.1,
  "chunks": [
    {
      "text": "chunk text...",
      "embedding": [0.012, -0.034, ...],
      "page": 5,
      "source": "document.pdf",
      "chunk_index": 3,
      "total_chunks": 42,
      "section_title": "Chapter 2"
    }
  ],
  "pages": [
    {"page": 1, "text": "full original page text..."},
    {"page": 2, "text": "..."}
  ]
}
```

---

## Setup

### Requirements

- Android Studio Hedgehog+ with JDK 17
- Git LFS (`git lfs install`) — required for model files
- GPU machine for the vectorizer server (see [vectorizer repo](https://github.com/shawnbarnes-lab/paperai-vectorizer-v2))

### Build

```bash
git lfs install
git clone https://github.com/shawnbarnes-lab/PaperAI.git
cd PaperAI
# Open in Android Studio → Build → Run
```

### Server Setup

```bash
git clone https://github.com/shawnbarnes-lab/paperai-vectorizer-v2.git
cd paperai-vectorizer-v2
pip install -r requirements.txt
python server.py
# Expose via ngrok: ngrok http 5000
```

In the app: Settings → Server URL → paste your ngrok URL.

---

## Device Compatibility

| Tier | Devices | Experience |
|------|---------|------------|
| **Full** | Galaxy S22 Ultra+, Pixel 8+ (8GB+ RAM) | LLM loads, AI summaries work |
| **Search only** | Galaxy Note 10, mid-range (4-6GB RAM) | LLM may not load, extractive fallback works, search is still fast |
| **Minimum** | Older devices (2-3GB RAM) | Search works, no AI answers |

The app gracefully degrades — search and source display work on any device, LLM is a bonus on capable hardware.

---

## Educational Use

Built as a teaching tool for the [Outschool AI Engineering course](https://outschool.com/classes/build-real-ai-on-real-hardware-a-teens-guide-to-generative-ai-11-hOmPLAiZ). Demonstrates RAG architecture, vector embeddings, on-device ML inference, and client-server AI systems.

---

## Related

- **Vectorizer Server:** [github.com/shawnbarnes-lab/paperai-vectorizer-v2](https://github.com/shawnbarnes-lab/paperai-vectorizer-v2)
- **Tensor Space:** [tensorspace.net](https://tensorspace.net)

---

## License

MIT

---

**Shawn Barnes** — Tensor Space LLC — [shawn.barnes@tensorspace.net](mailto:shawn.barnes@tensorspace.net)
