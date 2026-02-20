package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import io.objectbox.Box

/**
 * VectorStore manages document storage and similarity search using ObjectBox.
 */
class VectorStore {

    companion object {
        private const val TAG = "VectorStore"
    }

    private var documentBox: Box<DocumentChunk>? = null

    /**
     * Initialize the vector store using the existing BoxStore from ObjectBox singleton.
     */
    fun initialize(context: Context) {
        if (documentBox != null) {
            Log.d(TAG, "VectorStore already initialized")
            return
        }

        Log.d(TAG, "Getting ObjectBox from singleton...")

        // Use the existing BoxStore initialized in PaperAIApplication
        documentBox = ObjectBox.store.boxFor(DocumentChunk::class.java)
        Log.d(TAG, "VectorStore initialized with ${documentBox!!.count()} documents")
    }

    fun addDocument(chunk: DocumentChunk) {
        documentBox?.put(chunk)
    }

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
     * Get all chunks belonging to a specific document by source name.
     * Returns chunks sorted by page number, then by ID (document order).
     */
    fun getChunksBySource(sourceName: String): List<DocumentChunk> {
        val box = documentBox ?: return emptyList()
        
        val query = box.query(
            DocumentChunk_.sourceName.equal(sourceName)
        ).build()
        
        val results = query.find()
        query.close()
        
        Log.d(TAG, "Found ${results.size} chunks for document: $sourceName")
        
        // Sort by page number first, then by ID (chunk creation order = document order)
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
