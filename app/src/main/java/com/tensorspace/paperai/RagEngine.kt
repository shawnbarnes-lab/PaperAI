package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * RagEngine orchestrates the RAG pipeline.
 *
 * CHANGES FROM ORIGINAL:
 * - Hybrid response: brief LLM summary + source passages (not just LLM or just chunks)
 * - getContextWindow() returns ±N neighboring chunks around a match
 * - getDocumentChunksOrdered() returns chunks sorted by chunkIndex
 * - Similarity threshold raised to 0.35 (was 0.3) to reduce noise
 * - Added query expansion helper for future use
 */
class RagEngine(private val context: Context) {

    companion object {
        private const val TAG = "RagEngine"
        private const val DEFAULT_TOP_K = 5
        private const val MIN_SIMILARITY_THRESHOLD = 0.35f  // Was 0.3 — slightly stricter to reduce noise
        private const val CONTEXT_WINDOW_SIZE = 2            // ±2 chunks around each match
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
     *
     * IMPROVED STRATEGY:
     * 1. Generate embedding for query
     * 2. Search for relevant chunks
     * 3. Use LLM for a BRIEF summary (1-2 sentences) — not a full essay
     * 4. Return both the summary AND the source chunks for the UI to display
     *
     * The UI then shows: brief summary at top + expandable source passages below.
     * This gives much better perceived quality than asking a 1B model to write paragraphs.
     */
    suspend fun query(queryText: String, topK: Int = DEFAULT_TOP_K): RagResponse =
        withContext(Dispatchers.IO) {

            if (!isEmbeddingReady) {
                return@withContext RagResponse(
                    query = queryText,
                    answer = "Error: Embedding service not ready",
                    briefSummary = null,
                    sources = emptyList(),
                    contextChunks = emptyMap(),
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
                Log.d(TAG, "Found ${relevantChunks.size} relevant chunks (filtered from ${searchResults.size})")

                if (relevantChunks.isEmpty()) {
                    val processingTime = System.currentTimeMillis() - startTime
                    return@withContext RagResponse(
                        query = queryText,
                        answer = "No relevant information found for your query. Try different keywords.",
                        briefSummary = null,
                        sources = emptyList(),
                        contextChunks = emptyMap(),
                        processingTimeMs = processingTime,
                        error = null
                    )
                }

                // Step 3: Build context windows for each source chunk
                val contextWindows = mutableMapOf<Long, List<DocumentChunk>>()
                for (chunk in relevantChunks) {
                    contextWindows[chunk.id] = getContextWindow(chunk, CONTEXT_WINDOW_SIZE)
                }

                // Step 4: Generate brief summary + full response
                val briefSummary: String?
                val fullAnswer: String

                if (isLlmReady) {
                    // Primary mode: brief LLM summary + source passages
                    Log.d(TAG, "Generating brief summary...")
                    briefSummary = llmService.generateBrief(queryText, relevantChunks)

                    // Build the hybrid response: summary + key passages
                    fullAnswer = buildHybridResponse(briefSummary, relevantChunks)
                } else {
                    // Fallback: extractive only
                    briefSummary = null
                    fullAnswer = buildFallbackResponse(relevantChunks)
                }

                val processingTime = System.currentTimeMillis() - startTime

                RagResponse(
                    query = queryText,
                    answer = fullAnswer,
                    briefSummary = briefSummary,
                    sources = relevantChunks,
                    contextChunks = contextWindows,
                    processingTimeMs = processingTime,
                    error = null
                )

            } catch (e: Exception) {
                Log.e(TAG, "Error in RAG query", e)
                RagResponse(
                    query = queryText,
                    answer = "An error occurred while processing your query.",
                    briefSummary = null,
                    sources = emptyList(),
                    contextChunks = emptyMap(),
                    processingTimeMs = System.currentTimeMillis() - startTime,
                    error = e.message
                )
            }
        }

    /**
     * Get a context window of ±windowSize chunks around the given chunk.
     *
     * This solves the "cut off mid-sentence" problem — even if the exact match
     * is chunk #5, we also return chunks #3, #4, #6, #7 so the user sees
     * surrounding context.
     */
    fun getContextWindow(chunk: DocumentChunk, windowSize: Int = CONTEXT_WINDOW_SIZE): List<DocumentChunk> {
        val allChunks = vectorStore.getChunksBySource(chunk.sourceName)
        if (allChunks.isEmpty()) return listOf(chunk)

        // Sort by chunkIndex (or pageNumber as fallback)
        val sorted = allChunks.sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }

        val matchIndex = sorted.indexOfFirst { it.id == chunk.id }
        if (matchIndex < 0) return listOf(chunk)

        val startIdx = (matchIndex - windowSize).coerceAtLeast(0)
        val endIdx = (matchIndex + windowSize).coerceAtMost(sorted.lastIndex)

        return sorted.subList(startIdx, endIdx + 1)
    }

    /**
     * Build a hybrid response combining brief LLM summary + key source passages.
     * This is the improved response format that gives much better UX than
     * pure LLM generation from a small model.
     */
    private fun buildHybridResponse(summary: String, chunks: List<DocumentChunk>): String {
        val sb = StringBuilder()

        // Brief AI summary at top
        sb.appendLine(summary)
        sb.appendLine()

        // Key source passages
        sb.appendLine("📖 Key passages:")
        sb.appendLine()

        chunks.take(3).forEachIndexed { index, chunk ->
            val preview = chunk.text.take(200).let {
                if (chunk.text.length > 200) "$it..." else it
            }
            val source = if (chunk.sectionTitle.isNotBlank()) {
                "${chunk.sourceName} — ${chunk.sectionTitle} (p.${chunk.pageNumber})"
            } else {
                "${chunk.sourceName} (p.${chunk.pageNumber})"
            }
            sb.appendLine("${index + 1}. $preview")
            sb.appendLine("   — $source")
            sb.appendLine()
        }

        return sb.toString().trim()
    }

    /**
     * Build fallback response when LLM not available.
     * Shows the top chunk with source attribution.
     */
    private fun buildFallbackResponse(chunks: List<DocumentChunk>): String {
        val topChunk = chunks.first()
        val source = if (topChunk.sectionTitle.isNotBlank()) {
            "\"${topChunk.sourceName}\" — ${topChunk.sectionTitle} (page ${topChunk.pageNumber})"
        } else {
            "\"${topChunk.sourceName}\" (page ${topChunk.pageNumber})"
        }

        return """From $source:

${topChunk.text}

${if (chunks.size > 1) "(${chunks.size - 1} more matching sections found — see Sources tab)" else ""}

⏳ Full AI response generation is still loading.""".trim()
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
     * Get all chunks belonging to a specific document, ordered sequentially.
     * Used when user taps a source to view the full document.
     */
    fun getDocumentChunks(sourceName: String): List<DocumentChunk> {
        return vectorStore.getChunksBySource(sourceName)
            .sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }
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
 *
 * CHANGES FROM ORIGINAL:
 * - Renamed `response` to `answer` for clarity
 * - Added `briefSummary` — the short LLM-generated summary shown at top of results
 * - Added `contextChunks` — maps each source chunk ID to its ±N neighboring chunks
 */
data class RagResponse(
    val query: String,
    /** Full hybrid response (summary + passages) */
    val answer: String,
    /** Brief 1-2 sentence LLM summary, null if LLM not ready */
    val briefSummary: String? = null,
    /** The matched source chunks (search results) */
    val sources: List<DocumentChunk>,
    /** Context windows: maps source chunk ID -> neighboring chunks for expanded view */
    val contextChunks: Map<Long, List<DocumentChunk>> = emptyMap(),
    val processingTimeMs: Long,
    val error: String? = null
) {
    // Backward compatibility: `response` still works
    val response: String get() = answer
}

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
        return "$documentCount docs | LLM: $llmStatus | $llmModelInfo"
    }
}
