# paperAI - Android RAG Application

**Local-first document search powered by AI embeddings**

paperAI is an Android app that lets you search your personal documents using semantic search and AI. Upload PDFs, generate vector embeddings on your own GPU infrastructure, and search with natural language queries - all while keeping your data private.

---

## What This App Does

1. **User uploads documents** (PDF/TXT) to the app
2. **Documents are sent to vectorizer server** (running on your GPU machine)
3. **Server generates embeddings** using sentence-transformers on GPU
4. **User downloads .paperai file** (vector database)
5. **App searches locally** using semantic similarity
6. **LLM generates answers** from retrieved context (on-device)

**Key Feature:** Shows both AI-generated answers AND source text snippets side-by-side for verification.

---

## Architecture

```
┌─────────────────────────────────────────┐
│      Android App (This Repo)            │
│  ┌───────────────────────────────────┐  │
│  │  UI Layer                         │  │
│  │  - MainActivity.kt                │  │
│  │  - MainScreen.kt                  │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌───────────────────────────────────┐  │
│  │  Services                         │  │
│  │  - VectorizerClient.kt (HTTP)     │  │
│  │  - EmbeddingService.kt (on-device)│  │
│  │  - LlmService.kt (on-device)      │  │
│  │  - VectorSearchService.kt         │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌───────────────────────────────────┐  │
│  │  Data Layer                       │  │
│  │  - VectorStore.kt                 │  │
│  │  - DocumentChunk.kt               │  │
│  │  - ObjectBox (local DB)           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                │
                ▼ HTTPS (ngrok)
┌─────────────────────────────────────────┐
│   Vectorizer Server (Separate Repo)     │
│   github.com/shawnbarnes-lab/           │
│          paperai-vectorizer             │
│                                          │
│  - Flask REST API                       │
│  - sentence-transformers/all-MiniLM-L6  │
│  - Runs on RTX 3090/4090                │
│  - Exposed via ngrok tunnel             │
└─────────────────────────────────────────┘
```

---

## Requirements

### For Users (App)
- Android 8.0+ (API 26+)
- 4GB+ RAM recommended
- 500MB+ storage space
- Internet connection (only for vectorization, search is offline)

### For Developers
- Android Studio Hedgehog (2023.1.1) or newer
- JDK 17
- Gradle 8.0+
- Kotlin 1.9+
- Git LFS (for model files): https://git-lfs.github.com/

### For Backend (Separate Setup)
- Ubuntu 20.04+ with NVIDIA GPU
- See: https://github.com/shawnbarnes-lab/paperai-vectorizer

---

## Installation

### Option 1: Install from Release APK
1. Download latest `paperAI-v2.apk` from [Releases](https://github.com/shawnbarnes-lab/paperai-android/releases)
2. Enable "Install from Unknown Sources" in Android settings
3. Install APK
4. Open app → Settings → Enter vectorizer server URL

### Option 2: Build from Source

**Prerequisites:**
- Git LFS (required for model files): https://git-lfs.github.com/

```bash
# Install Git LFS (one time only)
git lfs install

# Clone repository (models download via Git LFS)
git clone https://github.com/shawnbarnes-lab/paperai-android.git
cd paperai-android

# Models are already included in paperai_models/ folder (via Git LFS)

# Open in Android Studio
# File → Open → Select 'paperai-android' folder

# Build
# Build → Make Project (Ctrl+F9)

# Run
# Run → Run 'app' (Shift+F10)
# Or: Build → Build Bundle(s) / APK(s) → Build APK(s)
```

---

## Configuration

### 1. Set Up Vectorizer Server

You need a GPU server running the vectorizer backend:
https://github.com/shawnbarnes-lab/paperai-vectorizer

Quick start:
```bash
# On your GPU machine
git clone https://github.com/shawnbarnes-lab/paperai-vectorizer.git
cd paperai-vectorizer
docker-compose up -d
ngrok http 5000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

### 2. Configure Android App

1. Open paperAI app
2. Go to Settings
3. Enter Server URL: `https://your-ngrok-url.ngrok.io`
4. Test connection (should show green checkmark)

---

## How to Use

### 1. Upload Documents
- Tap "Add Document" 
- Select PDF or TXT file
- Document is sent to your server for vectorization
- Download the `.paperai` file when ready

### 2. Load Vector Database
- Tap "Load Database"
- Select downloaded `.paperai` file
- App imports vectors into local ObjectBox database

### 3. Search
- Type natural language query in search box
- App finds semantically similar chunks
- Results show source text + page numbers

### 4. Generate AI Answers (Optional)
- Tap "Generate Answer" on search results
- On-device LLM creates summary from retrieved context
- See both AI answer AND source snippets for verification

---

## Key Components

### Android App Files

| File | Purpose |
|------|---------|
| `VectorizerClient.kt` | HTTP client for backend API |
| `EmbeddingService.kt` | On-device embedding model (all-MiniLM-L6-v2) |
| `LlmService.kt` | On-device LLM (Qwen2 1.5B) |
| `VectorSearchService.kt` | Cosine similarity search |
| `VectorStore.kt` | ObjectBox database wrapper |
| `MainActivity.kt` | Main UI entry point |
| `MainScreen.kt` | Jetpack Compose UI |

### Models Used

| Model | Size | Purpose | Location |
|-------|------|---------|----------|
| all-MiniLM-L6-v2 | 80MB | Text embeddings | On-device + Server |
| Qwen2-1.5B-4bit | 1.5GB | Answer generation | On-device (optional) |

**Models Included:** The all-MiniLM-L6-v2 model files (`model.onnx` and `tokenizer.json`) are included in this repository under `paperai_models/` via Git LFS. Install Git LFS before cloning to automatically download the models.

**Why same embedding model on both?**  
Server generates embeddings for documents. App generates embeddings for search queries. They must use the same model for compatibility.

---

## 🔐 Privacy & Security

- **All data stays private** - documents vectorized on YOUR infrastructure
- **No cloud services** - app works fully offline after vectorization
- **No tracking** - zero analytics or telemetry
- **Local storage** - vector databases stored on device only

**Note:** Server URL uses ngrok for convenience. For production, use proper HTTPS endpoint or VPN.

---

## Testing

### Unit Tests
```bash
./gradlew test
```

### Instrumented Tests
```bash
./gradlew connectedAndroidTest
```

### Manual Testing Checklist
- [ ] Upload 10-page PDF to server
- [ ] Download .paperai file
- [ ] Import to app successfully
- [ ] Search returns relevant results
- [ ] Source text matches page numbers
- [ ] (Optional) LLM generates coherent answer

---

## Troubleshooting

### "Cannot connect to server"
- Check server is running: `curl https://your-ngrok-url.ngrok.io/health`
- Verify ngrok tunnel is active
- Check app has internet permission
- Try different network (cellular vs WiFi)

### "Vectorization failed"
- Check server logs: `docker-compose logs`
- Ensure PDF is not corrupted or password-protected
- Verify GPU is available on server: `nvidia-smi`

### "Search returns no results"
- Check vector database is loaded (Settings → Database Status)
- Verify .paperai file was imported successfully
- Try broader search query

### "App crashes on search"
- Check ObjectBox is initialized properly
- Verify embedding model loaded correctly
- Check device has enough RAM (4GB+ recommended)

---

## Educational Use

This project was created as a teaching tool for **Outschool AI Engineering course: https://outschool.com/classes/build-real-ai-on-real-hardware-a-teens-guide-to-generative-ai-11-hOmPLAiZ**. It demonstrates:

- **RAG (Retrieval-Augmented Generation)** architecture
- **Vector embeddings** and semantic search
- **Client-server architecture** with REST APIs
- **On-device ML** inference on Android
- **GPU acceleration** for AI workloads
- **Docker containerization** for ML services

Perfect for students learning:
- AI engineering concepts
- Mobile app development
- Full-stack AI applications
- DevOps for ML systems

---

## Performance

**Tested on Samsung Galaxy S23 Ultra:**

| Operation | Time |
|-----------|------|
| Load 3,000 chunks | ~2 seconds |
| Search query | ~100-500ms |
| Generate embedding | ~50-100ms |
| LLM answer (optional) | ~15-60 seconds |

**Server (RTX 3090):**
| Operation | Time |
|-----------|------|
| 10-page PDF | ~5-10 seconds |
| 100-page PDF | ~30-60 seconds |
| 1000-page PDF | ~5-10 minutes |

---

## Roadmap

- [ ] Support more file types (DOCX, EPUB, Markdown)
- [ ] Batch document upload
- [ ] Export search results
- [ ] Dark mode UI
- [ ] Pre-built vector packs marketplace
- [ ] Multi-language support
- [ ] Offline vectorization (on-device)

---

## 🤝 Contributing

This is an educational project. Contributions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Related Projects

- **Vectorizer Server**: https://github.com/shawnbarnes-lab/paperai-vectorizer
- **tensorspace.net**: Local AI infrastructure consulting

---

## 📧 Contact

**Shawn Barnes**  
Tensor Space LLC  
Email: shawn.barnes@tensorspace.net  
GitHub: [@shawnbarnes-lab](https://github.com/shawnbarnes-lab)

---

## Acknowledgments

- **sentence-transformers** - Embedding models
- **Qwen2** - On-device LLM
- **ObjectBox** - Fast local database
- **Flask** - Python web framework
- **ngrok** - Secure tunnels for local servers

---

**Built with ❤️ for privacy-focused AI applications**
