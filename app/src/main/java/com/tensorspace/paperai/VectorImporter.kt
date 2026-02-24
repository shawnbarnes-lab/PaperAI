package com.tensorspace.paperai

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * VectorImporter loads pre-vectorized .paperai files into the app.
 *
 * v2.1 CHANGES:
 * - Parses new chunk fields: chunkIndex, totalChunks, sectionTitle
 * - Parses "pages" array with full original page text for document reconstruction
 * - Stores DocumentPage entities alongside DocumentChunk entities
 * - Backward compatible: v1 files without pages/new fields still import fine
 */
class VectorImporter(private val context: Context) {

    companion object {
        private const val TAG = "VectorImporter"
        private const val EXPECTED_EMBEDDING_DIM = 384
        private const val EXPECTED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    }

    /**
     * Result of an import operation.
     */
    data class ImportResult(
        val success: Boolean,
        val chunksImported: Int = 0,
        val pagesImported: Int = 0,
        val sourcesImported: Int = 0,
        val error: String? = null
    )

    /**
     * Import vectors from a .paperai file URI.
     */
    suspend fun importFromUri(uri: Uri, vectorStore: VectorStore): ImportResult =
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Importing from URI: $uri")

                val jsonString = context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream)).use { reader ->
                        reader.readText()
                    }
                } ?: return@withContext ImportResult(
                    success = false,
                    error = "Could not open file"
                )

                parseAndImport(jsonString, vectorStore)

            } catch (e: Exception) {
                Log.e(TAG, "Error importing from URI", e)
                ImportResult(
                    success = false,
                    error = "Import failed: ${e.message}"
                )
            }
        }

    /**
     * Import vectors from a .paperai file in assets.
     */
    suspend fun importFromAssets(assetPath: String, vectorStore: VectorStore): ImportResult =
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Importing from assets: $assetPath")

                val jsonString = context.assets.open(assetPath).use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream)).use { reader ->
                        reader.readText()
                    }
                }

                parseAndImport(jsonString, vectorStore)

            } catch (e: Exception) {
                Log.e(TAG, "Error importing from assets", e)
                ImportResult(
                    success = false,
                    error = "Import failed: ${e.message}"
                )
            }
        }

    /**
     * Import vectors from a JSON string.
     */
    suspend fun importFromString(jsonString: String, vectorStore: VectorStore): ImportResult =
        withContext(Dispatchers.IO) {
            parseAndImport(jsonString, vectorStore)
        }

    /**
     * Parse JSON and import chunks + pages into stores.
     */
    private fun parseAndImport(jsonString: String, vectorStore: VectorStore): ImportResult {
        try {
            val json = JSONObject(jsonString)

            // Version check — now supports up to 2.1
            val version = json.optDouble("version", 1.0)
            Log.d(TAG, "Vector file version: $version")

            // Validate embedding dimension
            val embeddingDim = json.optInt("embeddingDimension", EXPECTED_EMBEDDING_DIM)
            if (embeddingDim != EXPECTED_EMBEDDING_DIM) {
                return ImportResult(
                    success = false,
                    error = "Incompatible embedding dimension: $embeddingDim (expected $EXPECTED_EMBEDDING_DIM)"
                )
            }

            // Validate model (warning only)
            val model = json.optString("model", "")
            if (model.isNotEmpty() && model != EXPECTED_MODEL) {
                Log.w(TAG, "Vector file used different model: $model (expected $EXPECTED_MODEL)")
            }

            // --- Parse chunks ---
            val chunksArray = json.getJSONArray("chunks")
            val chunkCount = chunksArray.length()
            Log.d(TAG, "Parsing $chunkCount chunks...")

            var importedChunks = 0
            val sources = mutableSetOf<String>()

            for (i in 0 until chunkCount) {
                val chunkJson = chunksArray.getJSONObject(i)

                // Parse embedding
                val embeddingArray = chunkJson.getJSONArray("embedding")
                val embedding = FloatArray(embeddingArray.length()) { j ->
                    embeddingArray.getDouble(j).toFloat()
                }

                if (embedding.size != EXPECTED_EMBEDDING_DIM) {
                    Log.w(TAG, "Chunk $i has wrong embedding size: ${embedding.size}")
                    continue
                }

                // Create DocumentChunk with v2.1 fields (defaults for v1 files)
                val chunk = DocumentChunk(
                    text = chunkJson.getString("text"),
                    pageNumber = chunkJson.optInt("pageNumber", 0),
                    sourceName = chunkJson.optString("sourceName", "Unknown"),
                    embedding = embedding,
                    chunkIndex = chunkJson.optInt("chunkIndex", 0),
                    totalChunks = chunkJson.optInt("totalChunks", 0),
                    sectionTitle = chunkJson.optString("sectionTitle", ""),
                    originalFilePath = ""  // Not used yet
                )

                vectorStore.addDocument(chunk)
                importedChunks++
                sources.add(chunk.sourceName)

                if (importedChunks % 100 == 0) {
                    Log.d(TAG, "Imported $importedChunks / $chunkCount chunks...")
                }
            }

            // --- Parse pages (v2.1+ only) ---
            var importedPages = 0

            if (json.has("pages")) {
                val pagesArray = json.getJSONArray("pages")
                val pageCount = pagesArray.length()
                Log.d(TAG, "Parsing $pageCount pages for document reconstruction...")

                // Get the DocumentPage box from ObjectBox
                val pageBox = ObjectBox.store.boxFor(DocumentPage::class.java)

                // Determine source name for pages
                // Pages don't have sourceName in the JSON — derive from the chunks
                val primarySource = sources.firstOrNull() ?: "Unknown"

                // Remove any existing pages for this source (re-import safe)
                val existingPages = pageBox.query(
                    DocumentPage_.sourceName.equal(primarySource)
                ).build().find()
                if (existingPages.isNotEmpty()) {
                    pageBox.remove(existingPages)
                    Log.d(TAG, "Removed ${existingPages.size} existing pages for '$primarySource'")
                }

                // Import new pages
                val pagesToStore = mutableListOf<DocumentPage>()
                for (i in 0 until pageCount) {
                    val pageJson = pagesArray.getJSONObject(i)
                    val pageText = pageJson.optString("text", "")

                    if (pageText.isBlank()) continue

                    pagesToStore.add(
                        DocumentPage(
                            sourceName = primarySource,
                            pageNumber = pageJson.optInt("page", i + 1),
                            text = pageText
                        )
                    )
                }

                // Bulk insert
                pageBox.put(pagesToStore)
                importedPages = pagesToStore.size
                Log.d(TAG, "Imported $importedPages pages for '$primarySource'")
            } else {
                Log.d(TAG, "No pages array in file (v1 format) — document viewer will use chunks")
            }

            Log.d(TAG, "Import complete: $importedChunks chunks, $importedPages pages from ${sources.size} sources")

            return ImportResult(
                success = true,
                chunksImported = importedChunks,
                pagesImported = importedPages,
                sourcesImported = sources.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing vector file", e)
            return ImportResult(
                success = false,
                error = "Parse error: ${e.message}"
            )
        }
    }
}
