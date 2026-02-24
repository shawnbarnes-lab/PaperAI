package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import io.objectbox.Box

/**
 * VectorStore manages document storage and similarity search using ObjectBox.
 *
 * v2.1 CHANGES:
 * - Added searchDiversified() — limits results per source document so one
 *   massive document can't monopolize all result slots.
 *   Example: 259 survival manual chunks + 1 resume chunk — without diversity,
 *   all 5 results come from the manual. With maxPerSource=2, you get 2 manual
 *   results + the resume + others, giving a much better search experience.
 */
class VectorStore {

    companion object {
        private const val TAG = "VectorStore"
    }

    private var documentBox: Box<DocumentChunk>? = null

    fun initialize(context: Context) {
        if (documentBox != null) {
            Log.d(TAG, "VectorStore already initialized")
            return
        }

        Log.d(TAG, "Getting ObjectBox from singleton...")
        documentBox = ObjectBox.store.boxFor(DocumentChunk::class.java)
        Log.d(TAG, "VectorStore initialized with ${documentBox!!.count()} documents")
    }

    fun addDocument(chunk: DocumentChunk) {
        documentBox?.put(chunk)
    }

    /**
     * Basic search — returns top K nearest neighbors globally.
     * Use searchDiversified() instead for user-facing queries.
     */
    fun search(queryEmbedding: FloatArray, topK: Int): List<DocumentChunk> {
        val box = documentBox ?: return emptyList()

        if (box.count() == 0L) {
            Log.d(TAG, "No documents in store")
            return emptyList()
        }

        val query = box.query(
            DocumentChunk_.embedding.nearestNeighbors(queryEmbedding, topK)
        ).build()

        val results = query.findWithScores()
        query.close()

        Log.d(TAG, "Found ${results.size} results")

        return results.map { scoreResult ->
            val chunk = scoreResult.get()
            chunk.similarity = 1f - (scoreResult.score.toFloat() / 2f)
            chunk
        }
    }

    /**
     * Source-diversified search.
     *
     * Fetches a larger candidate pool (fetchMultiplier * topK), then limits
     * to maxPerSource results from any single document. Returns up to topK
     * results total, ranked by similarity.
     *
     * This prevents a 259-chunk document from taking all 5 slots when a
     * 1-chunk document is actually the best match for the query.
     *
     * @param queryEmbedding  The query vector
     * @param topK            How many results to return (default 5)
     * @param maxPerSource    Max results from any single sourceName (default 2)
     * @param fetchMultiplier How many extra candidates to fetch (default 4x)
     */
    fun searchDiversified(
        queryEmbedding: FloatArray,
        topK: Int = 5,
        maxPerSource: Int = 2,
        fetchMultiplier: Int = 4
    ): List<DocumentChunk> {
        val box = documentBox ?: return emptyList()

        if (box.count() == 0L) {
            Log.d(TAG, "No documents in store")
            return emptyList()
        }

        // Fetch a larger candidate pool
        val candidateCount = (topK * fetchMultiplier).coerceAtMost(box.count().toInt())
        val query = box.query(
            DocumentChunk_.embedding.nearestNeighbors(queryEmbedding, candidateCount)
        ).build()

        val results = query.findWithScores()
        query.close()

        // Score all candidates
        val scored = results.map { scoreResult ->
            val chunk = scoreResult.get()
            chunk.similarity = 1f - (scoreResult.score.toFloat() / 2f)
            chunk
        }

        // Apply source diversity: max N per document, keep sorted by similarity
        val sourceCounts = mutableMapOf<String, Int>()
        val diversified = mutableListOf<DocumentChunk>()

        for (chunk in scored) {
            val count = sourceCounts.getOrDefault(chunk.sourceName, 0)
            if (count < maxPerSource) {
                diversified.add(chunk)
                sourceCounts[chunk.sourceName] = count + 1
                if (diversified.size >= topK) break
            }
        }

        Log.d(TAG, "Diversified search: ${scored.size} candidates -> ${diversified.size} results " +
                "(${sourceCounts.size} sources, max $maxPerSource per source)")

        return diversified
    }

    /**
     * Get all chunks belonging to a specific document by source name.
     */
    fun getChunksBySource(sourceName: String): List<DocumentChunk> {
        val box = documentBox ?: return emptyList()

        val query = box.query(
            DocumentChunk_.sourceName.equal(sourceName)
        ).build()

        val results = query.find()
        query.close()

        Log.d(TAG, "Found ${results.size} chunks for document: $sourceName")

        return results.sortedWith(compareBy({ it.pageNumber }, { it.id }))
    }

    /**
     * Get all unique document names in the store.
     */
    fun getAllDocumentNames(): List<String> {
        val box = documentBox ?: return emptyList()

        return box.all
            .map { it.sourceName }
            .distinct()
            .sorted()
    }

    fun getDocumentCount(): Int = documentBox?.count()?.toInt() ?: 0

    fun clear() {
        documentBox?.removeAll()
        Log.d(TAG, "All documents cleared")
    }
}
