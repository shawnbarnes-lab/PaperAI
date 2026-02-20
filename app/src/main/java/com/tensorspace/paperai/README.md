# PaperAI Vectorizer

GPU-accelerated document vectorizer for PaperAI. Run the server on your powerful machine, upload documents from the Android app, and get vectors back automatically.

## Architecture

```
[PaperAI App]                    [Your RTX 3090 Server]
     │                                    │
     │  1. User picks document            │
     │  ─────────────────────────────►    │
     │     (upload .txt/.pdf)             │
     │                                    │
     │                           2. Vectorize with GPU
     │                                    │
     │  3. Receive .paperai vectors       │
     │  ◄─────────────────────────────    │
     │                                    │
     │  4. Auto-import into app           │
     ▼                                    │
  Ready to query!
```

## Server Setup (Ubuntu/RTX 3090)

```bash
# Install dependencies
pip install flask sentence-transformers PyPDF2 tqdm torch

# For CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run the server
python server.py

# Or specify host/port
python server.py --host 0.0.0.0 --port 5000
```

The server will show your local IP address - use this in the Android app.

## Android App Setup

1. Open PaperAI
2. Tap the **Settings** icon (gear) in the top bar
3. Enter your server URL (e.g., `http://192.168.1.100:5000`)
4. Tap **Test** to verify connection

## Usage

1. Tap the **+** button in PaperAI
2. Select a document (.txt or .pdf)
3. Wait for server to process (you'll see progress)
4. Document is automatically imported and ready to query!

## Alternative: Offline Vectorization

If you prefer to vectorize manually:

```bash
# Run on your GPU machine
python vectorizer.py survival_guide.pdf

# Transfer .paperai file to phone
# In app: tap "Import .paperai" instead of "Upload Document"
```

## Model Compatibility

Uses `sentence-transformers/all-MiniLM-L6-v2` - the same model as the Android app.
This ensures query embeddings on the phone match document embeddings from the server.

## Troubleshooting

**"Cannot connect to server"**
- Make sure server is running (`python server.py`)
- Check firewall allows port 5000
- Verify phone and server are on same network
- Try the server's local IP (not localhost)

**"Vectorization failed"**
- Check server console for error messages
- Ensure PDF is readable (not scanned image)
- Try with a .txt file first

## Android Permissions

Add to your AndroidManifest.xml:
```xml
<uses-permission android:name="android.permission.INTERNET" />
```
