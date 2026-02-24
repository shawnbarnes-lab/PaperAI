package com.tensorspace.paperai

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.VectorDistanceType

/**
 * DocumentChunk represents a single chunk of text from a document, along with its
 * vector embedding for similarity search.
 *
 * CHANGES FROM ORIGINAL:
 * - Added originalFilePath for opening the source PDF/TXT directly
 * - Added chunkIndex for ordering chunks within a document
 * - Added totalChunks so the UI knows the full document scope
 * - Added sectionTitle for better context display
 * - Similarity remains @Transient (not persisted, computed at search time)
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
     *
     * With the improved chunking strategy, chunks are now 400-500 words and
     * always break on paragraph boundaries — never mid-sentence.
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
     * Path to the original file in app-internal storage.
     * Used to open the actual PDF/TXT when the user taps "View Original".
     * Populated during import: e.g. "/data/data/com.tensorspace.paperai/files/docs/survival_guide.pdf"
     */
    var originalFilePath: String = "",

    /**
     * Zero-based index of this chunk within its source document.
     * Used for ordering chunks sequentially and for computing context windows (±N chunks).
     */
    var chunkIndex: Int = 0,

    /**
     * Total number of chunks in the source document.
     * Lets the UI show progress like "Section 3 of 47".
     */
    var totalChunks: Int = 0,

    /**
     * Section or chapter title extracted during chunking (if available).
     * For PDFs: heading text detected near the chunk.
     * For TXT: first line if it looks like a heading, otherwise empty.
     */
    var sectionTitle: String = "",

    /**
     * The vector embedding — 384-dim float array from all-MiniLM-L6-v2.
     *
     * @HnswIndex tells ObjectBox to build a special index (HNSW = Hierarchical
     * Navigable Small World) that makes vector searches extremely fast, even
     * with millions of vectors.
     *
     * Parameters:
     * - dimensions = 384: Must match the embedding model output size
     * - distanceType = COSINE: Standard for text similarity
     * - neighborsPerNode = 30: More = better accuracy, slower indexing
     * - indexingSearchCount = 200: Higher = better index quality, slower build
     */
    @HnswIndex(
        dimensions = 384,
        distanceType = VectorDistanceType.COSINE,
        neighborsPerNode = 30,
        indexingSearchCount = 200
    )
    var embedding: FloatArray = FloatArray(384),

    /**
     * Cosine similarity score computed at search time.
     * Not persisted to the database — only populated in search results.
     */
    @io.objectbox.annotation.Transient
    var similarity: Float = 0f
)
