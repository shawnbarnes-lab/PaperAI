package com.tensorspace.paperai

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * VectorSearchService finds document chunks that are semantically similar to a query.
 *
 * HOW VECTOR SEARCH WORKS:
 * ------------------------
 * Remember that embeddings place text in a 384-dimensional space where similar
 * meanings are close together? Vector search exploits this by:
 *
 * 1. Converting the user's search query into a vector (using EmbeddingService)
 * 2. Finding vectors in the database that are "close" to the query vector
 * 3. Returning the text chunks associated with those close vectors
 *
 * "Closeness" is measured using COSINE SIMILARITY:
 * - Two vectors pointing in the same direction have similarity = 1.0 (identical meaning)
 * - Two vectors pointing opposite directions have similarity = -1.0 (opposite meaning)
 * - Two perpendicular vectors have similarity = 0.0 (unrelated meaning)
 *
 * In practice, similarity scores for relevant results typically range from 0.3 to 0.9.
 * Scores below 0.2 usually indicate unrelated content.
 *
 * WHY THIS IS BETTER THAN KEYWORD SEARCH:
 * ---------------------------------------
 * Traditional search: "How to make fire" only matches documents containing those exact words.
 * Vector search: "How to make fire" also matches documents about "starting flames",
 *                "ignition techniques", "combustion methods", etc.
 *
 * The model understands that these concepts are related even though they use different words!
 */
class VectorSearchService(private val embeddingService: EmbeddingService) {

    companion object {
        private const val TAG = "VectorSearchService"

        // Default number of results to return
        const val DEFAULT_TOP_K = 5

        // Minimum similarity score to consider a result relevant
        // Results below this threshold are filtered out
        const val MIN_SIMILARITY_THRESHOLD = 0.1f
    }

    /**
     * Search for document chunks similar to the given query text.
     *
     * This is the main search function that paperAI users interact with.
     * It takes natural language queries and returns relevant source material.
     *
     * @param queryText The user's search query in natural language
     * @param topK How many results to return (default: 5)
     * @return List of SearchResult objects containing matching chunks and their scores
     */
    suspend fun search(queryText: String, topK: Int = DEFAULT_TOP_K): List<SearchResult> =
        withContext(Dispatchers.IO) {

            Log.d(TAG, "Searching for: '$queryText' (topK=$topK)")

            // Step 1: Convert the query text into an embedding vector
            val queryEmbedding = try {
                embeddingService.generateEmbedding(queryText)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate embedding for query", e)
                return@withContext emptyList()
            }

            Log.d(TAG, "Generated query embedding, searching database...")

            // Step 2: Search ObjectBox for similar vectors using HNSW index
            // ObjectBox's nearestNeighbors uses the HNSW index we defined on DocumentChunk
            val results = searchByVector(queryEmbedding, topK)

            Log.d(TAG, "Found ${results.size} results")
            results
        }

    /**
     * Search the database using a pre-computed embedding vector.
     *
     * This lower-level function is useful when you already have an embedding
     * and don't need to recompute it (e.g., for "find similar" functionality).
     *
     * @param queryEmbedding The 384-dimensional query vector
     * @param topK How many results to return
     * @return List of SearchResult objects sorted by similarity (highest first)
     */
    suspend fun searchByVector(queryEmbedding: FloatArray, topK: Int = DEFAULT_TOP_K): List<SearchResult> =
        withContext(Dispatchers.IO) {

            try {
                val chunkBox = ObjectBox.chunkBox
                val totalChunks = chunkBox.count()

                if (totalChunks == 0L) {
                    Log.d(TAG, "Database is empty, no results to return")
                    return@withContext emptyList()
                }

                Log.d(TAG, "Searching through $totalChunks chunks...")

                // Use ObjectBox's built-in nearest neighbor search
                // This leverages the HNSW index for fast approximate search
                val query = chunkBox.query(
                    DocumentChunk_.embedding.nearestNeighbors(queryEmbedding, topK)
                ).build()

                // findWithScores returns both the objects and their similarity scores
                val resultsWithScores = query.findWithScores()
                query.close()

                // Convert to our SearchResult format and filter by minimum similarity
                val searchResults = resultsWithScores
                    .filter { it.score >= MIN_SIMILARITY_THRESHOLD }
                    .map { scoredResult ->
                        SearchResult(
                            chunk = scoredResult.get(),
                            similarityScore = scoredResult.score.toFloat()
                        )
                    }
                    .sortedByDescending { it.similarityScore }

                Log.d(TAG, "Returning ${searchResults.size} results above threshold")
                searchResults

            } catch (e: Exception) {
                Log.e(TAG, "Error during vector search", e)
                emptyList()
            }
        }

    /**
     * Get the total number of document chunks in the database.
     * Useful for showing statistics to the user.
     */
    fun getTotalChunkCount(): Long {
        return try {
            ObjectBox.chunkBox.count()
        } catch (e: Exception) {
            Log.e(TAG, "Error getting chunk count", e)
            0L
        }
    }

    /**
     * Check if the database has any documents loaded.
     */
    fun hasDocuments(): Boolean {
        return getTotalChunkCount() > 0
    }

    /**
     * Get all unique source names (document titles) in the database.
     * Useful for showing users what documents they have loaded.
     */
    fun getLoadedSources(): List<String> {
        return try {
            ObjectBox.chunkBox.all
                .map { it.sourceName }
                .distinct()
                .sorted()
        } catch (e: Exception) {
            Log.e(TAG, "Error getting loaded sources", e)
            emptyList()
        }
    }
}

/**
 * SearchResult wraps a DocumentChunk with its similarity score.
 *
 * This is what we return from search operations. It pairs the actual content
 * (the chunk) with metadata about how well it matched (the score).
 *
 * @property chunk The matching document chunk with its text content
 * @property similarityScore How similar this chunk is to the query (0.0 to 1.0)
 *                          Higher scores mean more relevant results
 */
data class SearchResult(
    val chunk: DocumentChunk,
    val similarityScore: Float
) {
    /**
     * Format the similarity score as a percentage for display.
     * Example: 0.756 becomes "76%"
     */
    fun getScoreAsPercentage(): String {
        return "${(similarityScore * 100).toInt()}%"
    }

    /**
     * Get a human-readable relevance label based on the score.
     */
    fun getRelevanceLabel(): String {
        return when {
            similarityScore >= 0.8f -> "Highly Relevant"
            similarityScore >= 0.6f -> "Very Relevant"
            similarityScore >= 0.4f -> "Relevant"
            similarityScore >= 0.2f -> "Somewhat Relevant"
            else -> "Low Relevance"
        }
    }
}