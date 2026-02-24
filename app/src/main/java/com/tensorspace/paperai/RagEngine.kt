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
 * - getDocumentPages() returns full original page text for document reconstruction
 */
class RagEngine(private val context: Context) {

    companion object {
        private const val TAG = "RagEngine"
        private const val DEFAULT_TOP_K = 5
        private const val MIN_SIMILARITY_THRESHOLD = 0.35f
        private const val CONTEXT_WINDOW_SIZE = 2
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

        isEmbeddingReady = try {
            embeddingService.initialize()
            Log.d(TAG, "Embedding service ready")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize embedding service", e)
            false
        }

        vectorStore.initialize(context)
        Log.d(TAG, "Vector store ready with ${vectorStore.getDocumentCount()} documents")

        isLlmReady = try {
            val result = llmService.initialize()
            Log.d(TAG, "LLM service ready: $result")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM service", e)
            false
        }

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
                Log.d(TAG, "Generating embedding for: '$queryText'")
                val queryEmbedding = embeddingService.generateEmbedding(queryText)

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

                val contextWindows = mutableMapOf<Long, List<DocumentChunk>>()
                for (chunk in relevantChunks) {
                    contextWindows[chunk.id] = getContextWindow(chunk, CONTEXT_WINDOW_SIZE)
                }

                val briefSummary: String?
                val fullAnswer: String

                if (isLlmReady) {
                    Log.d(TAG, "Generating brief summary...")
                    briefSummary = llmService.generateBrief(queryText, relevantChunks)
                    fullAnswer = buildHybridResponse(briefSummary, relevantChunks)
                } else {
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
     */
    fun getContextWindow(chunk: DocumentChunk, windowSize: Int = CONTEXT_WINDOW_SIZE): List<DocumentChunk> {
        val allChunks = vectorStore.getChunksBySource(chunk.sourceName)
        if (allChunks.isEmpty()) return listOf(chunk)

        val sorted = allChunks.sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }

        val matchIndex = sorted.indexOfFirst { it.id == chunk.id }
        if (matchIndex < 0) return listOf(chunk)

        val startIdx = (matchIndex - windowSize).coerceAtLeast(0)
        val endIdx = (matchIndex + windowSize).coerceAtMost(sorted.lastIndex)

        return sorted.subList(startIdx, endIdx + 1)
    }

    private fun buildHybridResponse(summary: String, chunks: List<DocumentChunk>): String {
        val sb = StringBuilder()

        sb.appendLine(summary)
        sb.appendLine()
        sb.appendLine("\uD83D\uDCD6 Key passages:")
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

    suspend fun generateEmbedding(text: String): FloatArray = withContext(Dispatchers.IO) {
        embeddingService.generateEmbedding(text)
    }

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
     */
    fun getDocumentChunks(sourceName: String): List<DocumentChunk> {
        return vectorStore.getChunksBySource(sourceName)
            .sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }
    }

    /**
     * Get full original page text for a document.
     * Used by the document viewer to reconstruct the complete document
     * and highlight matched chunks within it.
     *
     * Returns pages sorted by page number.
     * Returns empty list for v1 documents that don't have page data.
     */
    fun getDocumentPages(sourceName: String): List<DocumentPage> {
        val pageBox = ObjectBox.store.boxFor(DocumentPage::class.java)
        return pageBox.query(
            DocumentPage_.sourceName.equal(sourceName)
        ).build().find().sortedBy { it.pageNumber }
    }

    fun getVectorStore(): VectorStore = vectorStore

    fun getStats(): RagStats {
        return RagStats(
            documentCount = vectorStore.getDocumentCount(),
            isEmbeddingReady = isEmbeddingReady,
            isLlmReady = isLlmReady,
            llmModelInfo = llmService.getModelInfo()
        )
    }

    fun clearDocuments() {
        vectorStore.clear()
        try {
            val pageBox = ObjectBox.store.boxFor(DocumentPage::class.java)
            pageBox.removeAll()
            Log.d(TAG, "Documents and pages cleared")
        } catch (e: Exception) {
            Log.w(TAG, "Could not clear pages: ${e.message}")
        }
    }

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
    val answer: String,
    val briefSummary: String? = null,
    val sources: List<DocumentChunk>,
    val contextChunks: Map<Long, List<DocumentChunk>> = emptyMap(),
    val processingTimeMs: Long,
    val error: String? = null
) {
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
