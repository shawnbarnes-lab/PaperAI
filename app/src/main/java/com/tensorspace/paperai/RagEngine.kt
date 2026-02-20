package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * RagEngine orchestrates the RAG pipeline.
 */
class RagEngine(private val context: Context) {

    companion object {
        private const val TAG = "RagEngine"
        private const val DEFAULT_TOP_K = 5
        private const val MIN_SIMILARITY_THRESHOLD = 0.3f
    }

    private val embeddingService = EmbeddingService(context)
    private val llmService = LlmService(context)
    private val vectorStore = VectorStore()

    private var isEmbeddingReady = false
    private var isLlmReady = false

    /**
     * Initialize all RAG components.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "Initializing RAG engine...")

        // Initialize embedding service
        isEmbeddingReady = try {
            embeddingService.initialize()
            Log.d(TAG, "Embedding service ready")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize embedding service", e)
            false
        }

        // Initialize vector store
        vectorStore.initialize(context)
        Log.d(TAG, "Vector store ready with ${vectorStore.getDocumentCount()} documents")

        // Initialize LLM service (can be slow)
        isLlmReady = try {
            val result = llmService.initialize()
            Log.d(TAG, "LLM service ready: $result")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM service", e)
            false
        }

        // Return true if at least embedding is ready (search will work)
        isEmbeddingReady
    }

    /**
     * Perform a RAG query.
     */
    suspend fun query(queryText: String, topK: Int = DEFAULT_TOP_K): RagResponse =
        withContext(Dispatchers.IO) {

            if (!isEmbeddingReady) {
                return@withContext RagResponse(
                    query = queryText,
                    response = "Error: Embedding service not ready",
                    sources = emptyList(),
                    processingTimeMs = 0,
                    error = "Embedding service not initialized"
                )
            }

            val startTime = System.currentTimeMillis()

            try {
                // Step 1: Generate embedding for query
                Log.d(TAG, "Generating embedding for: '$queryText'")
                val queryEmbedding = embeddingService.generateEmbedding(queryText)

                // Step 2: Search for relevant chunks
                Log.d(TAG, "Searching for top $topK chunks")
                val searchResults = vectorStore.search(queryEmbedding, topK)
                val relevantChunks = searchResults.filter { it.similarity >= MIN_SIMILARITY_THRESHOLD }
                Log.d(TAG, "Found ${relevantChunks.size} relevant chunks")

                // Step 3: Generate response with LLM
                val response = if (isLlmReady && relevantChunks.isNotEmpty()) {
                    Log.d(TAG, "Generating LLM response...")
                    llmService.generate(queryText, relevantChunks)
                } else if (relevantChunks.isNotEmpty()) {
                    // Fallback if LLM not ready
                    buildFallbackResponse(relevantChunks)
                } else {
                    "No relevant information found for your query."
                }

                val processingTime = System.currentTimeMillis() - startTime

                RagResponse(
                    query = queryText,
                    response = response,
                    sources = relevantChunks,
                    processingTimeMs = processingTime,
                    error = null
                )

            } catch (e: Exception) {
                Log.e(TAG, "Error in RAG query", e)
                RagResponse(
                    query = queryText,
                    response = "An error occurred while processing your query.",
                    sources = emptyList(),
                    processingTimeMs = System.currentTimeMillis() - startTime,
                    error = e.message
                )
            }
        }

    /**
     * Generate embedding for text.
     */
    suspend fun generateEmbedding(text: String): FloatArray = withContext(Dispatchers.IO) {
        embeddingService.generateEmbedding(text)
    }

    /**
     * Load documents into the vector store.
     */
    suspend fun loadDocuments(chunks: List<DocumentChunk>): Int = withContext(Dispatchers.IO) {
        Log.d(TAG, "Loading ${chunks.size} document chunks...")
        var count = 0
        for (chunk in chunks) {
            vectorStore.addDocument(chunk)
            count++
        }
        Log.d(TAG, "Loaded $count documents")
        count
    }

    /**
     * Get all chunks belonging to a specific document.
     * Used when user taps a source to view the full document.
     */
    fun getDocumentChunks(sourceName: String): List<DocumentChunk> {
        return vectorStore.getChunksBySource(sourceName)
    }
    
    /**
     * Get the vector store for direct operations (like import).
     */
    fun getVectorStore(): VectorStore = vectorStore

    /**
     * Get stats about the current state.
     */
    fun getStats(): RagStats {
        return RagStats(
            documentCount = vectorStore.getDocumentCount(),
            isEmbeddingReady = isEmbeddingReady,
            isLlmReady = isLlmReady,
            llmModelInfo = llmService.getModelInfo()
        )
    }

    /**
     * Clear all documents.
     */
    fun clearDocuments() {
        vectorStore.clear()
        Log.d(TAG, "Documents cleared")
    }

    /**
     * Build fallback response when LLM not available.
     */
    private fun buildFallbackResponse(chunks: List<DocumentChunk>): String {
        val topChunk = chunks.first()
        return """Based on "${topChunk.sourceName}" (page ${topChunk.pageNumber}):

${topChunk.text}

(Note: Full AI response generation is still loading.)"""
    }

    /**
     * Release resources.
     */
    fun close() {
        embeddingService.close()
        llmService.close()
        Log.d(TAG, "RAG engine closed")
    }
}

/**
 * Response from a RAG query.
 */
data class RagResponse(
    val query: String,
    val response: String,
    val sources: List<DocumentChunk>,
    val processingTimeMs: Long,
    val error: String? = null
)

/**
 * Stats about the RAG engine state.
 */
data class RagStats(
    val documentCount: Int,
    val isEmbeddingReady: Boolean,
    val isLlmReady: Boolean,
    val llmModelInfo: String
) {
    fun hasDocuments(): Boolean = documentCount > 0

    fun getSummary(): String {
        val llmStatus = if (isLlmReady) "ready" else "loading"
        return "$documentCount docs | LLM: $llmStatus"
    }
}
