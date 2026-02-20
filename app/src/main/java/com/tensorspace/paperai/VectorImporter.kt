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
 * This allows documents to be processed on a powerful GPU machine
 * and then loaded into the phone for offline RAG queries.
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
        val sourcesImported: Int = 0,
        val error: String? = null
    )
    
    /**
     * Import vectors from a .paperai file URI.
     * Use this when importing via Android file picker.
     */
    suspend fun importFromUri(uri: Uri, vectorStore: VectorStore): ImportResult = 
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Importing from URI: $uri")
                
                // Read the file content
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
     * Use this for bundled vector files.
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
     * Parse JSON and import chunks into vector store.
     */
    private fun parseAndImport(jsonString: String, vectorStore: VectorStore): ImportResult {
        try {
            val json = JSONObject(jsonString)
            
            // Validate version
            val version = json.optInt("version", 1)
            if (version > 1) {
                Log.w(TAG, "Vector file version $version may not be fully compatible")
            }
            
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
            
            // Parse chunks
            val chunksArray = json.getJSONArray("chunks")
            val chunkCount = chunksArray.length()
            
            Log.d(TAG, "Parsing $chunkCount chunks...")
            
            var importedCount = 0
            val sources = mutableSetOf<String>()
            
            for (i in 0 until chunkCount) {
                val chunkJson = chunksArray.getJSONObject(i)
                
                // Parse embedding array
                val embeddingArray = chunkJson.getJSONArray("embedding")
                val embedding = FloatArray(embeddingArray.length()) { j ->
                    embeddingArray.getDouble(j).toFloat()
                }
                
                // Validate embedding size
                if (embedding.size != EXPECTED_EMBEDDING_DIM) {
                    Log.w(TAG, "Chunk $i has wrong embedding size: ${embedding.size}")
                    continue
                }
                
                // Create DocumentChunk
                val chunk = DocumentChunk(
                    text = chunkJson.getString("text"),
                    pageNumber = chunkJson.optInt("pageNumber", 0),
                    sourceName = chunkJson.optString("sourceName", "Unknown"),
                    embedding = embedding
                )
                
                // Add to vector store
                vectorStore.addDocument(chunk)
                importedCount++
                sources.add(chunk.sourceName)
                
                // Log progress for large imports
                if (importedCount % 100 == 0) {
                    Log.d(TAG, "Imported $importedCount / $chunkCount chunks...")
                }
            }
            
            Log.d(TAG, "Import complete: $importedCount chunks from ${sources.size} sources")
            
            return ImportResult(
                success = true,
                chunksImported = importedCount,
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
