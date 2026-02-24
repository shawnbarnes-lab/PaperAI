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
 * v2.1 PROMPT OVERHAUL:
 * - Temperature 0.3 → 0.1 (near-greedy for factual RAG on a 1B model)
 * - TOP_K 20 → 10 (tighter sampling = less hallucination)
 * - Source names now included in prompt so model can reference them
 * - Forced response prefix on ALL prompts (not just full)
 * - Brief prompt is primary path — asks for direct answer, not "read then answer"
 * - Extractive fallback searches ALL chunks, not just the first one
 * - Better garbage detection: catches repetition loops + off-topic rambling
 */
class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MAX_TOKENS = 1024
        // Near-greedy decoding for factual RAG — 1B models hallucinate badly with any creativity
        private const val TEMPERATURE = 0.1f
        private const val TOP_K = 10
        private const val TOP_P = 0.8f
        private const val MAX_CONTEXT_CHARS = 1500
        private const val MAX_CHUNK_CHARS = 400
    }

    enum class ModelTier(
        val displayName: String,
        val assetPath: String,
        val fileName: String,
        val minRamMb: Long,
        val priority: Int
    ) {
        GEMMA_1B(
            displayName = "Gemma 3 1B (int4)",
            assetPath = "models/gemma3.task",
            fileName = "gemma3.task",
            minRamMb = 2048,
            priority = 1
        ),
        // Future: uncomment when bundling a bigger model
        // GEMMA_3N_E2B(
        //     displayName = "Gemma 3n E2B",
        //     assetPath = "models/gemma3n-e2b.litertlm",
        //     fileName = "gemma3n-e2b.litertlm",
        //     minRamMb = 3072,
        //     priority = 2
        // ),
    }

    private var llmInference: LlmInference? = null
    private var isInitialized = false
    private var modelPath: String? = null
    private var activeModel: ModelTier? = null

    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) {
            Log.d(TAG, "LLM already initialized")
            return@withContext true
        }

        try {
            Log.d(TAG, "Starting LLM initialization...")

            val selectedModel = selectBestModel()
            if (selectedModel == null) {
                Log.e(TAG, "No compatible model found for this device")
                return@withContext false
            }
            activeModel = selectedModel
            Log.d(TAG, "Selected model: ${selectedModel.displayName}")

            val modelFile = copyModelFromAssets(selectedModel)
            if (modelFile == null) {
                Log.e(TAG, "Failed to copy model from assets")
                return@withContext false
            }
            modelPath = modelFile.absolutePath
            Log.d(TAG, "Model file ready at: $modelPath")

            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath!!)
                .setMaxTokens(MAX_TOKENS)
                .setPreferredBackend(LlmInference.Backend.CPU)
                .build()

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

    private fun selectBestModel(): ModelTier? {
        val runtime = Runtime.getRuntime()
        val availableRamMb = runtime.maxMemory() / (1024 * 1024)
        Log.d(TAG, "Available RAM: ${availableRamMb}MB")

        return ModelTier.values()
            .filter { tier ->
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

    private fun copyModelFromAssets(model: ModelTier): File? {
        try {
            val modelFile = File(context.filesDir, model.fileName)

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
     * Generate a FULL response. Used for detailed answers.
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
     * Generate a BRIEF 1-2 sentence summary — the primary RAG mode.
     *
     * A 1B model produces much better results when asked for a short direct
     * answer vs. a long synthesized essay. The UI shows this summary + the
     * actual source passages below it.
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

                Log.d(TAG, "Brief response in ${elapsed}ms: '${response.take(100)}'")

                val cleaned = cleanResponse(response)

                // Detect garbage: too short, too long, or repetition loops
                when {
                    cleaned.length < 10 -> {
                        Log.w(TAG, "Brief response too short (${cleaned.length} chars), extractive fallback")
                        buildExtractiveAnswer(query, contextChunks)
                    }
                    cleaned.length > 500 -> {
                        Log.w(TAG, "Brief response too long (${cleaned.length} chars), extractive fallback")
                        buildExtractiveAnswer(query, contextChunks)
                    }
                    hasRepetitionLoop(cleaned) -> {
                        Log.w(TAG, "Brief response has repetition loop, extractive fallback")
                        buildExtractiveAnswer(query, contextChunks)
                    }
                    else -> cleaned
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error in brief generation, falling back to extractive", e)
                buildExtractiveAnswer(query, contextChunks)
            }
        }

    /**
     * Streaming generation (token by token for UI feedback).
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
     * Key design for 1B models:
     * - Source names labeled so model can reference them
     * - Minimal instructions — every extra word degrades quality
     * - Forced response prefix steers generation immediately
     * - Hard caps prevent token overflow
     */
    private fun buildRagPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(3).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(40)
            val truncatedText = chunk.text.take(MAX_CHUNK_CHARS)
            "[Source: $label]\n$truncatedText"
        }.joinToString("\n---\n")

        val finalContext = if (contextText.length > MAX_CONTEXT_CHARS) {
            contextText.take(MAX_CONTEXT_CHARS)
        } else {
            contextText
        }

        return """<start_of_turn>user
Answer ONLY from these sources. Be brief and factual.

$finalContext

Question: $query
<end_of_turn>
<start_of_turn>model
Answer: """
    }

    /**
     * Build the brief prompt — primary RAG mode.
     *
     * For 1B models, the key tricks:
     * 1. Put the question FIRST so the model sees it before the context
     *    (attention is strongest at start and end of context)
     * 2. Include source names so the answer can reference them
     * 3. Forced prefix "Based on [source]," steers the model to cite
     * 4. Absolute minimum instructions — no "read the sources then..."
     */
    private fun buildBriefPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        // Take top 2 chunks with source labels
        val contextText = contextChunks.take(2).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(40)
            "[${label}] ${chunk.text.take(300)}"
        }.joinToString("\n---\n")

        // Build a smart prefix using the top source name
        val topSource = contextChunks.first().sourceName
            .replace(".pdf", "")
            .replace("_", " ")
            .take(30)

        return """<start_of_turn>user
Question: $query

Sources:
$contextText

Answer in 1-2 sentences only.
<end_of_turn>
<start_of_turn>model
Based on $topSource, """
    }

    /**
     * Pure extractive fallback — no LLM needed.
     *
     * IMPROVED: Now searches ALL chunks (not just first) for the best
     * matching sentence. With diversified search results, we might have
     * chunks from 3 different documents — the best sentence could be
     * in any of them.
     */
    private fun buildExtractiveAnswer(query: String, contextChunks: List<DocumentChunk>): String {
        if (contextChunks.isEmpty()) return "No relevant information found."

        val queryWords = query.lowercase().split(Regex("\\s+")).toSet()

        // Search ALL chunks for the best matching sentence, not just the first
        var bestSentence: String? = null
        var bestScore = -1
        var bestSource: String? = null

        for (chunk in contextChunks.take(5)) {
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
            // Last resort: just show beginning of top chunk
            val topChunk = contextChunks.first()
            topChunk.text.take(200).trim() + "..."
        }
    }

    /**
     * Detect repetition loops — a common failure mode for small models.
     * If any 8+ word phrase repeats 3+ times, it's a loop.
     */
    private fun hasRepetitionLoop(text: String): Boolean {
        val words = text.split(Regex("\\s+"))
        if (words.size < 24) return false

        // Check for repeated 8-word sequences
        val windowSize = 8
        val sequences = mutableMapOf<String, Int>()

        for (i in 0..words.size - windowSize) {
            val seq = words.subList(i, i + windowSize).joinToString(" ").lowercase()
            val count = (sequences[seq] ?: 0) + 1
            sequences[seq] = count
            if (count >= 3) return true
        }

        return false
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
            .replace(Regex("^\\s*Answer:\\s*"), "")
            .replace(Regex("^\\s*Based on the sources:\\s*"), "")
            .trim()
    }

    fun isReady(): Boolean = isInitialized && llmInference != null

    fun getModelInfo(): String {
        return if (isInitialized && activeModel != null) {
            activeModel!!.displayName
        } else {
            "Loading..."
        }
    }

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
