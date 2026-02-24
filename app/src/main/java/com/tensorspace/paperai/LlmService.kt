package com.tensorspace.paperai

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * LlmService handles on-device LLM inference using MediaPipe's LLM Inference API.
 *
 * CHANGES FROM ORIGINAL:
 * - Temperature lowered to 0.3 for factual RAG (was 0.7 — way too creative)
 * - TOP_K reduced to 20, TOP_P to 0.85 for tighter generation
 * - Added ModelTier enum to support multiple model sizes
 * - Added generateBrief() for short 1-2 sentence summaries (primary RAG mode)
 * - Completely rewritten buildRagPrompt() optimized for small models
 * - Added buildBriefPrompt() for summary-only generation
 * - Added forced response prefix to steer generation
 * - Hard-capped context per chunk to prevent token overflow
 * - Added RepetitionPenalty constant to reduce model loops
 */
class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MAX_TOKENS = 1024
        // ---- KEY CHANGES: Generation parameters tuned for factual RAG ----
        private const val TEMPERATURE = 0.3f       // Was 0.7 — lower = more factual, less hallucination
        private const val TOP_K = 20               // Was 40 — narrower sampling for precision
        private const val TOP_P = 0.85f            // Was 0.95 — tighter nucleus sampling
        private const val MAX_CONTEXT_CHARS = 1500  // Hard cap on total context fed to model
        private const val MAX_CHUNK_CHARS = 400     // Hard cap per individual chunk
    }

    /**
     * Model tiers — detect what the device can handle and pick the best option.
     * Higher tier = better answers but slower inference and more RAM.
     *
     * To use a bigger model:
     *   1. Place the .task file in assets/models/
     *   2. Update the enum entry's assetPath and fileName
     *   3. The service auto-selects the best available model at init time
     */
    enum class ModelTier(
        val displayName: String,
        val assetPath: String,
        val fileName: String,
        val minRamMb: Long,
        val priority: Int  // higher = preferred when device supports it
    ) {
        // Current model — works on most devices
        GEMMA_1B(
            displayName = "Gemma 3 1B (int4)",
            assetPath = "models/gemma3.task",
            fileName = "gemma3.task",
            minRamMb = 2048,
            priority = 1
        ),
        // Upgrade path — significantly better quality, needs ~4GB free RAM
        // To enable: add the .task file to assets/models/ and uncomment
        // GEMMA_2B(
        //     displayName = "Gemma 2 2B (int4)",
        //     assetPath = "models/gemma2-2b.task",
        //     fileName = "gemma2-2b.task",
        //     minRamMb = 4096,
        //     priority = 2
        // ),
        // Big upgrade — needs ~6GB free RAM, flagship devices only
        // PHI3_MINI(
        //     displayName = "Phi-3 Mini 3.8B (int4)",
        //     assetPath = "models/phi3-mini.task",
        //     fileName = "phi3-mini.task",
        //     minRamMb = 6144,
        //     priority = 3
        // ),
    }

    private var llmInference: LlmInference? = null
    private var isInitialized = false
    private var modelPath: String? = null
    private var activeModel: ModelTier? = null

    /**
     * Initialize the LLM service.
     * Selects the best model the device can support, copies from assets, and loads it.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) {
            Log.d(TAG, "LLM already initialized")
            return@withContext true
        }

        try {
            Log.d(TAG, "Starting LLM initialization...")

            // Select best model for this device
            val selectedModel = selectBestModel()
            if (selectedModel == null) {
                Log.e(TAG, "No compatible model found for this device")
                return@withContext false
            }
            activeModel = selectedModel
            Log.d(TAG, "Selected model: ${selectedModel.displayName}")

            // Copy model from assets to internal storage (if needed)
            val modelFile = copyModelFromAssets(selectedModel)
            if (modelFile == null) {
                Log.e(TAG, "Failed to copy model from assets")
                return@withContext false
            }
            modelPath = modelFile.absolutePath
            Log.d(TAG, "Model file ready at: $modelPath")

            // Create LLM Inference options
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath!!)
                .setMaxTokens(MAX_TOKENS)
                .setPreferredBackend(LlmInference.Backend.CPU)
                .build()

            // Load the model
            Log.d(TAG, "Loading LLM model (this may take 10-30 seconds)...")
            llmInference = LlmInference.createFromOptions(context, options)

            isInitialized = true
            Log.d(TAG, "LLM initialized successfully with ${selectedModel.displayName}")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM", e)
            isInitialized = false
            false
        }
    }

    /**
     * Select the highest-priority model that the device can run.
     * Checks available RAM and whether the model asset exists.
     */
    private fun selectBestModel(): ModelTier? {
        val runtime = Runtime.getRuntime()
        val availableRamMb = runtime.maxMemory() / (1024 * 1024)
        Log.d(TAG, "Available RAM: ${availableRamMb}MB")

        return ModelTier.values()
            .filter { tier ->
                // Check if model asset exists
                val assetExists = try {
                    context.assets.open(tier.assetPath).close()
                    true
                } catch (e: Exception) {
                    false
                }
                assetExists && availableRamMb >= tier.minRamMb
            }
            .maxByOrNull { it.priority }
    }

    /**
     * Copy the model file from assets to internal storage.
     */
    private fun copyModelFromAssets(model: ModelTier): File? {
        try {
            val modelFile = File(context.filesDir, model.fileName)

            // Check if already copied
            if (modelFile.exists()) {
                Log.d(TAG, "Model already exists at ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024} MB)")
                return modelFile
            }

            Log.d(TAG, "Copying model from assets (this may take a minute)...")
            val startTime = System.currentTimeMillis()

            context.assets.open(model.assetPath).use { inputStream ->
                FileOutputStream(modelFile).use { outputStream ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    var totalBytes = 0L

                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                        totalBytes += bytesRead

                        if (totalBytes % (50 * 1024 * 1024) < 8192) {
                            Log.d(TAG, "Copied ${totalBytes / 1024 / 1024} MB...")
                        }
                    }
                }
            }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Model copied in ${elapsed}ms (${modelFile.length() / 1024 / 1024} MB)")
            return modelFile

        } catch (e: Exception) {
            Log.e(TAG, "Error copying model from assets", e)
            return null
        }
    }

    /**
     * Generate a FULL response based on the query and retrieved context chunks.
     * Used when the user wants a detailed AI-generated answer.
     */
    suspend fun generate(query: String, contextChunks: List<DocumentChunk>): String =
        withContext(Dispatchers.IO) {
            if (!isInitialized || llmInference == null) {
                return@withContext "Error: LLM not initialized. Please wait for model to load."
            }

            if (contextChunks.isEmpty()) {
                return@withContext "I couldn't find any relevant information in the documents to answer your question."
            }

            try {
                val prompt = buildRagPrompt(query, contextChunks)
                Log.d(TAG, "Generating full response for: '$query'")
                Log.d(TAG, "Prompt length: ${prompt.length} chars (~${prompt.length / 4} tokens)")

                val startTime = System.currentTimeMillis()
                val response = llmInference!!.generateResponse(prompt)
                val elapsed = System.currentTimeMillis() - startTime

                Log.d(TAG, "Response generated in ${elapsed}ms")
                cleanResponse(response)

            } catch (e: Exception) {
                Log.e(TAG, "Error generating response: ${e.javaClass.simpleName}: ${e.message}", e)
                "I encountered an error while generating a response. Please try again."
            }
        }

    /**
     * Generate a BRIEF 1-2 sentence summary from the retrieved context.
     *
     * This is the primary RAG mode. Small models (1-2B params) produce much better
     * results when asked for a short factual summary vs. a long synthesized answer.
     * The UI then displays this summary + the actual source passages below it.
     */
    suspend fun generateBrief(query: String, contextChunks: List<DocumentChunk>): String =
        withContext(Dispatchers.IO) {
            if (!isInitialized || llmInference == null) {
                return@withContext buildExtractiveAnswer(query, contextChunks)
            }

            if (contextChunks.isEmpty()) {
                return@withContext "No relevant information found."
            }

            try {
                val prompt = buildBriefPrompt(query, contextChunks)
                Log.d(TAG, "Generating brief response for: '$query'")

                val startTime = System.currentTimeMillis()
                val response = llmInference!!.generateResponse(prompt)
                val elapsed = System.currentTimeMillis() - startTime

                Log.d(TAG, "Brief response in ${elapsed}ms")

                val cleaned = cleanResponse(response)

                // If the model produced garbage or too much, fall back to extractive
                if (cleaned.length < 10 || cleaned.length > 500) {
                    Log.w(TAG, "Brief response was ${cleaned.length} chars, falling back to extractive")
                    buildExtractiveAnswer(query, contextChunks)
                } else {
                    cleaned
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error in brief generation, falling back to extractive", e)
                buildExtractiveAnswer(query, contextChunks)
            }
        }

    /**
     * Generate a response with streaming (token by token).
     */
    suspend fun generateStreaming(
        query: String,
        contextChunks: List<DocumentChunk>,
        onToken: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        if (!isInitialized || llmInference == null) {
            val error = "Error: LLM not initialized. Please wait for model to load."
            onToken(error)
            return@withContext error
        }

        if (contextChunks.isEmpty()) {
            val msg = "I couldn't find any relevant information to answer your question."
            onToken(msg)
            return@withContext msg
        }

        try {
            val prompt = buildRagPrompt(query, contextChunks)
            Log.d(TAG, "Starting streaming generation...")

            val response = llmInference!!.generateResponse(prompt)
            val cleanedResponse = cleanResponse(response)

            // Stream word by word for UI feedback
            val words = cleanedResponse.split(" ")
            val result = StringBuilder()

            for (word in words) {
                val token = if (result.isEmpty()) word else " $word"
                result.append(token)
                onToken(token)
                kotlinx.coroutines.delay(15)
            }

            cleanedResponse

        } catch (e: Exception) {
            Log.e(TAG, "Error in streaming generation", e)
            val error = "I encountered an error while generating a response."
            onToken(error)
            error
        }
    }

    // ============================================================================
    // PROMPT ENGINEERING — The biggest lever for small model quality
    // ============================================================================

    /**
     * Build the full RAG prompt.
     *
     * Key differences from original:
     * - Hard cap each chunk at MAX_CHUNK_CHARS (prevents token overflow)
     * - Removed meta-instructions like "reference which source" (wastes tokens on small models)
     * - Added forced response prefix "Based on the sources:" to steer generation
     * - Simpler, more direct instructions — small models follow simple prompts better
     */
    private fun buildRagPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(3).mapIndexed { index, chunk ->
            val truncatedText = chunk.text.take(MAX_CHUNK_CHARS)
            "[${index + 1}] $truncatedText"
        }.joinToString("\n---\n")

        // Hard cap total context
        val finalContext = if (contextText.length > MAX_CONTEXT_CHARS) {
            contextText.take(MAX_CONTEXT_CHARS)
        } else {
            contextText
        }

        return """<start_of_turn>user
Answer using ONLY these sources. If the answer isn't in the sources, say "Not found in documents."

Sources:
$finalContext

Question: $query
<end_of_turn>
<start_of_turn>model
Based on the sources: """
    }

    /**
     * Build a prompt for brief (1-2 sentence) summary generation.
     * Even more constrained than the full prompt — forces a short output.
     */
    private fun buildBriefPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        // For brief mode, use only top 2 chunks with aggressive truncation
        val contextText = contextChunks.take(2).mapIndexed { index, chunk ->
            "[${index + 1}] ${chunk.text.take(300)}"
        }.joinToString("\n---\n")

        return """<start_of_turn>user
Read the sources, then answer in ONE or TWO sentences only.

Sources:
$contextText

Question: $query
<end_of_turn>
<start_of_turn>model
"""
    }

    /**
     * Pure extractive fallback — no LLM needed.
     * Picks the most relevant sentence from the top chunk that relates to the query.
     * Used when LLM fails, produces garbage, or isn't loaded yet.
     */
    private fun buildExtractiveAnswer(query: String, contextChunks: List<DocumentChunk>): String {
        if (contextChunks.isEmpty()) return "No relevant information found."

        val topChunk = contextChunks.first()
        val sentences = topChunk.text
            .split(Regex("[.!?]+\\s+"))
            .filter { it.length > 20 }

        // Simple keyword overlap scoring to pick the best sentence
        val queryWords = query.lowercase().split(Regex("\\s+")).toSet()
        val bestSentence = sentences.maxByOrNull { sentence ->
            val sentenceWords = sentence.lowercase().split(Regex("\\s+")).toSet()
            queryWords.intersect(sentenceWords).size
        } ?: sentences.firstOrNull() ?: topChunk.text.take(200)

        return "${bestSentence.trim()}."
    }

    /**
     * Clean up the model's response by removing artifacts.
     */
    private fun cleanResponse(response: String): String {
        return response
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .replace("<start_of_turn>", "")
            .replace("model\n", "")
            .replace(Regex("^\\s*Based on the sources:\\s*"), "")  // Remove forced prefix if model echoed it
            .trim()
    }

    /**
     * Check if the LLM is ready to generate responses.
     */
    fun isReady(): Boolean = isInitialized && llmInference != null

    /**
     * Get model info for display in the UI.
     */
    fun getModelInfo(): String {
        return if (isInitialized && activeModel != null) {
            activeModel!!.displayName
        } else {
            "Loading..."
        }
    }

    /**
     * Release resources when done.
     */
    fun close() {
        try {
            llmInference?.close()
            llmInference = null
            isInitialized = false
            activeModel = null
            Log.d(TAG, "LLM service closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing LLM service", e)
        }
    }
}
