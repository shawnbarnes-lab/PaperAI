package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * RagEngine orchestrates the RAG pipeline.
 *
 * v3.0 — Gemma 2B + 3-layer grounding
 * - Similarity threshold at 0.50
 * - Keyword relevance filter
 * - Post-generation grounding check tuned for 2B model (45%)
 * - Extractive fallback for when LLM still hallucinates
 * - Refusal phrase detection
 */
class RagEngine(private val context: Context) {

    companion object {
        private const val TAG = "RagEngine"
        private const val DEFAULT_TOP_K = 5
        private const val MIN_SIMILARITY_THRESHOLD = 0.50f
        private const val KEYWORD_OVERLAP_THRESHOLD = 1
        private const val CONTEXT_WINDOW_SIZE = 2
        private const val MAX_RESULTS_PER_SOURCE = 2
        private const val GROUNDING_THRESHOLD = 0.45f
    }

    private val embeddingService = EmbeddingService(context)
    private val llmService = LlmService(context)
    private val vectorStore = VectorStore()

    private var isEmbeddingReady = false
    private var isLlmReady = false

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
     * Perform a RAG query with source-diversified search and relevance filtering.
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

                Log.d(TAG, "Searching for top $topK chunks (max $MAX_RESULTS_PER_SOURCE per source)")
                val searchResults = vectorStore.searchDiversified(
                    queryEmbedding = queryEmbedding,
                    topK = topK,
                    maxPerSource = MAX_RESULTS_PER_SOURCE
                )

                // LAYER 1: Similarity threshold
                val similarChunks = searchResults.filter { it.similarity >= MIN_SIMILARITY_THRESHOLD }
                Log.d(TAG, "Similarity filter: ${searchResults.size} → ${similarChunks.size} chunks (threshold=$MIN_SIMILARITY_THRESHOLD)")

                // LAYER 2: Keyword relevance check
                val relevantChunks = similarChunks.filter { chunk ->
                    hasKeywordRelevance(queryText, chunk.text)
                }
                Log.d(TAG, "Keyword filter: ${similarChunks.size} → ${relevantChunks.size} chunks")

                if (similarChunks.size > relevantChunks.size) {
                    val filtered = similarChunks - relevantChunks.toSet()
                    filtered.forEach { chunk ->
                        Log.d(TAG, "Filtered out (no keyword match): '${chunk.text.take(60)}...' sim=${chunk.similarity}")
                    }
                }

                if (relevantChunks.isEmpty()) {
                    val processingTime = System.currentTimeMillis() - startTime

                    val bestScore = searchResults.maxByOrNull { it.similarity }?.similarity ?: 0f
                    Log.d(TAG, "No relevant results. Best similarity was $bestScore (need >$MIN_SIMILARITY_THRESHOLD + keyword match)")

                    return@withContext RagResponse(
                        query = queryText,
                        answer = "This topic is not covered in the loaded documents. Try a question related to the documents you've imported.",
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
                    Log.d(TAG, "Generating brief summary with ${relevantChunks.size} grounded chunks...")
                    val rawSummary = llmService.generateBrief(queryText, relevantChunks)

                    // LAYER 3: Post-generation grounding check
                    briefSummary = if (isResponseGrounded(rawSummary, relevantChunks)) {
                        rawSummary
                    } else {
                        Log.w(TAG, "LLM response failed grounding check, using extractive fallback")
                        buildExtractiveAnswer(queryText, relevantChunks)
                    }

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
     * Check if the query and chunk share meaningful keyword overlap.
     */
    private fun hasKeywordRelevance(query: String, chunkText: String): Boolean {
        val stopWords = setOf(
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "don", "now", "i", "me", "my", "you", "your", "it", "its", "we",
            "they", "them", "what", "which", "who", "whom", "this", "that"
        )

        val queryWords = query.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), "")
            .split(Regex("\\s+"))
            .filter { it.length > 2 && it !in stopWords }
            .toSet()

        val chunkWords = chunkText.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), "")
            .split(Regex("\\s+"))
            .filter { it.length > 2 && it !in stopWords }
            .toSet()

        val overlap = queryWords.intersect(chunkWords)
        val hasOverlap = overlap.size >= KEYWORD_OVERLAP_THRESHOLD

        if (!hasOverlap) {
            Log.d(TAG, "No keyword overlap. Query words: $queryWords, Chunk sample: ${chunkWords.take(10)}")
        }

        return hasOverlap
    }

    /**
     * Verify that the LLM's response is actually grounded in the source chunks.
     * 2B model is better at paraphrasing so we allow 45% threshold.
     */
    private fun isResponseGrounded(response: String, chunks: List<DocumentChunk>): Boolean {
        val stopWords = setOf(
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "and", "but", "or", "not",
            "this", "that", "it", "its", "you", "your", "i", "me", "my",
            "based", "according", "sources", "information", "document",
            "must", "use", "need", "using", "also", "very", "just", "only",
            "loaded", "documents", "covered", "topic", "contain"
        )

        // If the model correctly refused, that's grounded
        val refusalPhrases = listOf(
            "not covered in the loaded documents",
            "do not contain information",
            "not covered in the documents",
            "no information about this",
            "not covered in the loaded",
            "documents do not contain"
        )
        if (refusalPhrases.any { response.lowercase().contains(it) }) {
            Log.d(TAG, "Grounding check: model correctly refused (grounded)")
            return true
        }

        val responseWords = response.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), "")
            .split(Regex("\\s+"))
            .filter { it.length > 3 && it !in stopWords }
            .toSet()

        if (responseWords.isEmpty()) return true

        val chunkWords = chunks.flatMap { chunk ->
            chunk.text.lowercase()
                .replace(Regex("[^a-z0-9\\s]"), "")
                .split(Regex("\\s+"))
                .filter { it.length > 3 }
        }.toSet()

        val groundedWords = responseWords.intersect(chunkWords)
        val ungroundedWords = responseWords - chunkWords
        val groundingRatio = if (responseWords.isNotEmpty()) {
            groundedWords.size.toFloat() / responseWords.size.toFloat()
        } else {
            1.0f
        }

        Log.d(TAG, "Grounding check: ${groundedWords.size}/${responseWords.size} words grounded (${(groundingRatio * 100).toInt()}%)")
        Log.d(TAG, "Ungrounded words: $ungroundedWords")

        return groundingRatio >= GROUNDING_THRESHOLD
    }

    /**
     * Pure extractive answer — pull the best matching sentence directly from chunks.
     */
    private fun buildExtractiveAnswer(query: String, chunks: List<DocumentChunk>): String {
        if (chunks.isEmpty()) return "No relevant information found."

        val queryWords = query.lowercase().split(Regex("\\s+")).toSet()

        var bestSentence: String? = null
        var bestScore = -1
        var bestSource: String? = null

        for (chunk in chunks.take(5)) {
            val sentences = chunk.text
                .split(Regex("[.!?]+\\s+"))
                .filter { it.length > 20 }

            for (sentence in sentences) {
                val sentenceWords = sentence.lowercase().split(Regex("\\s+")).toSet()
                val score = queryWords.intersect(sentenceWords).size
                if (score > bestScore) {
                    bestScore = score
                    bestSentence = sentence.trim()
                    bestSource = chunk.sourceName
                        .replace(".pdf", "")
                        .replace("_", " ")
                }
            }
        }

        return if (bestSentence != null && bestSource != null) {
            "From $bestSource: $bestSentence."
        } else {
            val topChunk = chunks.first()
            topChunk.text.take(200).trim() + "..."
        }
    }

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

    fun getDocumentChunks(sourceName: String): List<DocumentChunk> {
        return vectorStore.getChunksBySource(sourceName)
            .sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }
    }

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