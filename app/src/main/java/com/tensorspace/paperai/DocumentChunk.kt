package com.tensorspace.paperai

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.VectorDistanceType

/**
 * DocumentChunk represents a single chunk of text from a document, along with its
 * vector embedding for similarity search.
 *
 * In RAG (Retrieval Augmented Generation), we don't store entire documents as-is.
 * Instead, we break them into smaller "chunks" (typically 200-500 words each).
 * Each chunk gets converted into a vector (a list of 384 numbers) that captures
 * its semantic meaning. When a user searches, we convert their query to a vector
 * and find chunks with similar vectors.
 *
 * The @Entity annotation tells ObjectBox this class should be stored in the database.
 * ObjectBox will automatically create a table for DocumentChunk objects.
 */
@Entity
data class DocumentChunk(
    /**
     * Unique identifier for this chunk. ObjectBox requires every entity to have
     * an @Id field of type Long. Setting default to 0 tells ObjectBox to
     * auto-generate the ID when we insert a new chunk.
     */
    @Id
    var id: Long = 0,

    /**
     * The actual text content of this chunk. This is what we show to the user
     * as "source material" and what we feed to the LLM as context.
     */
    var text: String = "",

    /**
     * Which page (or section) this chunk came from in the original document.
     * Useful for showing users where to find more information.
     */
    var pageNumber: Int = 0,

    /**
     * The name or title of the source document this chunk came from.
     * Example: "SAS Survival Handbook" or "Wilderness Medicine Guide"
     */
    var sourceName: String = "",

    /**
     * The vector embedding - this is the magic that makes semantic search work!
     *
     * The all-MiniLM-L6-v2 model converts text into 384 numbers (dimensions).
     * These numbers encode the "meaning" of the text in a way that similar
     * texts have similar numbers. We can then use cosine similarity to find
     * chunks that are semantically similar to a user's query.
     *
     * @HnswIndex tells ObjectBox to build a special index (HNSW = Hierarchical
     * Navigable Small World) that makes vector searches extremely fast, even
     * with millions of vectors. Without this index, we'd have to compare against
     * every single vector, which would be slow.
     *
     * Parameters explained:
     * - dimensions = 384: The size of our vectors (must match the embedding model)
     * - distanceType = COSINE: How to measure similarity (cosine is standard for text)
     * - neighborsPerNode = 30: HNSW tuning - more neighbors = better accuracy, slower indexing
     * - indexingSearchCount = 200: HNSW tuning - higher = better index quality, slower build
     */
    @HnswIndex(
        dimensions = 384,
        distanceType = VectorDistanceType.COSINE,
        neighborsPerNode = 30,
        indexingSearchCount = 200
    )
    var embedding: FloatArray = FloatArray(384),

    @io.objectbox.annotation.Transient
    var similarity: Float = 0f
)