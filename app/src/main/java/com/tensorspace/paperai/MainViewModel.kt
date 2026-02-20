package com.tensorspace.paperai

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * MainViewModel manages all UI state for the main screen of paperAI.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainViewModel"
        private const val PREFS_NAME = "paperai_prefs"
        private const val KEY_SERVER_URL = "vectorizer_server_url"
        private const val DEFAULT_SERVER_URL = "https://unvaulting-untangentially-jenice.ngrok-free.dev/"
    }

    private val ragEngine = RagEngine(application)
    private val vectorImporter = VectorImporter(application)
    private val vectorizerClient = VectorizerClient(application)
    private val prefs = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    private val _selectedTab = MutableStateFlow(0)
    val selectedTab: StateFlow<Int> = _selectedTab.asStateFlow()
    
    // Import result for showing feedback
    private val _importResult = MutableStateFlow<VectorImporter.ImportResult?>(null)
    val importResult: StateFlow<VectorImporter.ImportResult?> = _importResult.asStateFlow()
    
    // Server URL for vectorizer
    private val _serverUrl = MutableStateFlow(prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL)
    val serverUrl: StateFlow<String> = _serverUrl.asStateFlow()
    
    // Upload progress
    private val _uploadProgress = MutableStateFlow<UploadProgress?>(null)
    val uploadProgress: StateFlow<UploadProgress?> = _uploadProgress.asStateFlow()
    
    // Server status
    private val _serverStatus = MutableStateFlow<VectorizerClient.ServerStatus?>(null)
    val serverStatus: StateFlow<VectorizerClient.ServerStatus?> = _serverStatus.asStateFlow()

    init {
        Log.d(TAG, "MainViewModel created, initializing RAG engine...")
        initializeEngine()
    }

    fun onSearchQueryChanged(query: String) {
        _searchQuery.value = query
    }

    fun onSearch() {
        val query = _searchQuery.value.trim()

        if (query.isBlank()) {
            Log.d(TAG, "Search attempted with empty query, ignoring")
            return
        }

        Log.d(TAG, "Executing search for: '$query'")

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isSearching = true,
                error = null
            )

            try {
                val response = ragEngine.query(query)

                _uiState.value = _uiState.value.copy(
                    isSearching = false,
                    lastResponse = response,
                    error = response.error
                )

                Log.d(TAG, "Search completed: ${response.sources.size} sources, " +
                        "${response.processingTimeMs}ms")

            } catch (e: Exception) {
                Log.e(TAG, "Search failed", e)
                _uiState.value = _uiState.value.copy(
                    isSearching = false,
                    error = "Search failed: ${e.message}"
                )
            }
        }
    }

    fun onTabSelected(index: Int) {
        _selectedTab.value = index
    }

    fun onClearSearch() {
        _searchQuery.value = ""
        _uiState.value = _uiState.value.copy(
            lastResponse = null,
            error = null
        )
    }

    /**
     * Get all chunks belonging to a specific document.
     * Used when user taps a source card to view the full document.
     */
    fun getDocumentChunks(sourceName: String): List<DocumentChunk> {
        return ragEngine.getDocumentChunks(sourceName)
    }

    /**
     * Import vectors from a .paperai file URI.
     */
    fun importVectors(uri: android.net.Uri) {
        viewModelScope.launch {
            Log.d(TAG, "Importing vectors from: $uri")
            
            _uiState.value = _uiState.value.copy(isLoading = true)
            _importResult.value = null
            
            try {
                val result = vectorImporter.importFromUri(uri, ragEngine.getVectorStore())
                _importResult.value = result
                
                if (result.success) {
                    val stats = ragEngine.getStats()
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        stats = stats
                    )
                    Log.d(TAG, "Import successful: ${result.chunksImported} chunks from ${result.sourcesImported} sources")
                } else {
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = result.error
                    )
                    Log.e(TAG, "Import failed: ${result.error}")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Import error", e)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Import failed: ${e.message}"
                )
                _importResult.value = VectorImporter.ImportResult(
                    success = false,
                    error = e.message
                )
            }
        }
    }
    
    /**
     * Clear the import result (dismiss the feedback).
     */
    fun clearImportResult() {
        _importResult.value = null
    }
    
    /**
     * Update the vectorizer server URL.
     */
    fun setServerUrl(url: String) {
        _serverUrl.value = url
        prefs.edit().putString(KEY_SERVER_URL, url).apply()
        Log.d(TAG, "Server URL updated: $url")
    }
    
    /**
     * Check if the vectorizer server is online.
     */
    fun checkServer() {
        viewModelScope.launch {
            Log.d(TAG, "Checking server: ${_serverUrl.value}")
            _serverStatus.value = vectorizerClient.checkServer(_serverUrl.value)
        }
    }
    
    /**
     * Upload a document to the server, vectorize it, and auto-import.
     * This is the main workflow for adding new documents.
     */
    fun uploadAndVectorize(documentUri: android.net.Uri) {
        viewModelScope.launch {
            Log.d(TAG, "Starting upload and vectorize: $documentUri")
            
            _uploadProgress.value = UploadProgress(
                inProgress = true,
                progress = 0f,
                status = "Starting..."
            )
            
            try {
                // Send to server for vectorization
                val result = vectorizerClient.vectorizeDocument(
                    serverUrl = _serverUrl.value,
                    documentUri = documentUri,
                    onProgress = { progress, status ->
                        _uploadProgress.value = UploadProgress(
                            inProgress = true,
                            progress = progress,
                            status = status
                        )
                    }
                )
                
                if (result.success && result.vectorJson != null) {
                    // Auto-import the vectors
                    _uploadProgress.value = UploadProgress(
                        inProgress = true,
                        progress = 0.95f,
                        status = "Importing vectors..."
                    )
                    
                    val importResult = vectorImporter.importFromString(
                        result.vectorJson,
                        ragEngine.getVectorStore()
                    )
                    
                    _uploadProgress.value = null
                    
                    if (importResult.success) {
                        val stats = ragEngine.getStats()
                        _uiState.value = _uiState.value.copy(stats = stats)
                        _importResult.value = importResult
                        Log.d(TAG, "Upload and import complete: ${importResult.chunksImported} chunks")
                    } else {
                        _importResult.value = importResult
                        Log.e(TAG, "Import failed: ${importResult.error}")
                    }
                    
                } else {
                    _uploadProgress.value = null
                    _importResult.value = VectorImporter.ImportResult(
                        success = false,
                        error = result.error ?: "Vectorization failed"
                    )
                    Log.e(TAG, "Vectorization failed: ${result.error}")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Upload error", e)
                _uploadProgress.value = null
                _importResult.value = VectorImporter.ImportResult(
                    success = false,
                    error = "Upload failed: ${e.message}"
                )
            }
        }
    }
    
    /**
     * Cancel ongoing upload (clears progress state).
     */
    fun cancelUpload() {
        _uploadProgress.value = null
    }

    fun loadSampleDocuments() {
        viewModelScope.launch {
            Log.d(TAG, "Loading sample documents...")

            _uiState.value = _uiState.value.copy(isLoading = true)

            try {
                val sampleChunks = createSampleDocuments()
                val count = ragEngine.loadDocuments(sampleChunks)
                val stats = ragEngine.getStats()

                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    stats = stats
                )

                Log.d(TAG, "Loaded $count sample documents")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load sample documents", e)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Failed to load documents: ${e.message}"
                )
            }
        }
    }

    fun clearDocuments() {
        viewModelScope.launch {
            Log.d(TAG, "Clearing documents...")

            ragEngine.clearDocuments()

            val stats = ragEngine.getStats()
            _uiState.value = _uiState.value.copy(
                stats = stats,
                lastResponse = null
            )

            Log.d(TAG, "Documents cleared")
        }
    }

    fun retryInitialization() {
        initializeEngine()
    }

    private fun initializeEngine() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isInitializing = true,
                error = null
            )

            try {
                val success = ragEngine.initialize()

                if (success) {
                    val stats = ragEngine.getStats()
                    _uiState.value = _uiState.value.copy(
                        isInitializing = false,
                        isEngineReady = true,
                        stats = stats
                    )
                    Log.d(TAG, "RAG engine initialized successfully")
                } else {
                    _uiState.value = _uiState.value.copy(
                        isInitializing = false,
                        isEngineReady = false,
                        error = "Failed to initialize AI engine"
                    )
                    Log.e(TAG, "RAG engine initialization returned false")
                }

            } catch (e: Exception) {
                Log.e(TAG, "RAG engine initialization threw exception", e)
                _uiState.value = _uiState.value.copy(
                    isInitializing = false,
                    isEngineReady = false,
                    error = "Initialization error: ${e.message}"
                )
            }
        }
    }

    /**
     * Create sample documents for testing the RAG pipeline.
     * Now generates REAL embeddings using the ONNX model.
     */
    private suspend fun createSampleDocuments(): List<DocumentChunk> {
        val chunks = mutableListOf<DocumentChunk>()

        val survivalContent = listOf(
            Triple(
                "Starting a fire is one of the most critical survival skills. The bow drill method is effective in most environments. You need a fireboard, spindle, bow, and socket. Create a notch in the fireboard, place tinder beneath it, and use the bow to spin the spindle rapidly until you create an ember.",
                1,
                "SAS Survival Handbook"
            ),
            Triple(
                "Finding clean water is essential for survival. Look for flowing water rather than stagnant pools. Water can be purified by boiling for at least one minute. In emergency situations, you can collect dew from plants in the early morning using a cloth.",
                2,
                "SAS Survival Handbook"
            ),
            Triple(
                "Building an emergency shelter should be a priority. A debris hut can be constructed using fallen branches and leaves. The shelter should be just large enough to fit your body to retain heat. Always insulate the ground beneath you to prevent heat loss.",
                3,
                "SAS Survival Handbook"
            ),
            Triple(
                "Navigation without a compass is possible using natural signs. The sun rises in the east and sets in the west. At night, the North Star indicates true north in the Northern Hemisphere. Moss tends to grow on the north side of trees in northern latitudes.",
                4,
                "Wilderness Navigation Guide"
            ),
            Triple(
                "First aid for bleeding wounds requires direct pressure. Apply firm pressure with a clean cloth for at least 10 minutes. Elevate the wound above heart level if possible. For severe bleeding, a tourniquet may be necessary as a last resort.",
                5,
                "Wilderness First Aid Manual"
            ),
            Triple(
                "Edible plants can provide nutrition in survival situations. Learn to identify common edible plants in your area before an emergency. The universal edibility test involves touching the plant to your skin, then lips, then tongue over several hours to check for reactions.",
                6,
                "Foraging Field Guide"
            ),
            Triple(
                "Signaling for rescue increases your chances of being found. Three of anything is a universal distress signal - three fires, three whistle blasts, or three gunshots. Use mirrors or reflective surfaces to signal aircraft. Create large ground-to-air signals visible from above.",
                7,
                "Search and Rescue Handbook"
            ),
            Triple(
                "Hypothermia occurs when body temperature drops below 95°F (35°C). Early symptoms include shivering, confusion, and slurred speech. Treatment involves removing wet clothing, insulating the person, and providing warm fluids if conscious. Never rub frostbitten skin.",
                8,
                "Wilderness First Aid Manual"
            )
        )

        // Create chunks with REAL embeddings from the ONNX model
        for ((text, page, source) in survivalContent) {
            Log.d(TAG, "Generating embedding for: $source page $page")

            // Generate REAL embedding using the ONNX model
            val embeddingArray = ragEngine.generateEmbedding(text)

            chunks.add(
                DocumentChunk(
                    text = text,
                    pageNumber = page,
                    sourceName = source,
                    embedding = embeddingArray
                )
            )
        }

        return chunks
    }

    override fun onCleared() {
        super.onCleared()
        Log.d(TAG, "MainViewModel being cleared, releasing resources...")
        ragEngine.close()
    }
}

data class UiState(
    val isInitializing: Boolean = true,
    val isEngineReady: Boolean = false,
    val isLoading: Boolean = false,
    val isSearching: Boolean = false,
    val lastResponse: RagResponse? = null,
    val stats: RagStats? = null,
    val error: String? = null
) {
    fun shouldShowLoading(): Boolean = isInitializing || isLoading || isSearching

    fun canSearch(): Boolean = isEngineReady && !isSearching && !isInitializing

    fun getStatusMessage(): String {
        return when {
            isInitializing -> "Initializing AI engine..."
            error != null -> "Error: $error"
            isSearching -> "Searching..."
            isLoading -> "Loading documents..."
            !isEngineReady -> "Engine not ready"
            stats?.hasDocuments() == true -> stats.getSummary()
            else -> "No documents loaded"
        }
    }
}

/**
 * Progress state for document upload/vectorization.
 */
data class UploadProgress(
    val inProgress: Boolean = false,
    val progress: Float = 0f,
    val status: String = ""
)
