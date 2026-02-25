package com.tensorspace.paperai

import android.app.ActivityManager
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
 * v3.0 — Gemma 2B int4
 * - Upgraded from 1B to 2B for much better instruction following
 * - Better source citation and paraphrasing
 * - Can actually refuse when sources don't contain an answer
 * - Tuned prompts for 2B model's stronger capabilities
 */
class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MAX_TOKENS = 1024
        private const val MAX_CONTEXT_CHARS = 2000
        private const val MAX_CHUNK_CHARS = 500
        private const val MODEL_ASSET_PATH = "models/gemma-2b-int4.task"
        private const val MODEL_FILE_NAME = "gemma-2b-int4.task"
        private const val MODEL_DISPLAY_NAME = "Gemma 2B (int4)"
        private const val MIN_RAM_MB = 4096L
    }

    private var llmInference: LlmInference? = null
    private var isInitialized = false
    private var modelPath: String? = null

    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) {
            Log.d(TAG, "LLM already initialized")
            return@withContext true
        }

        try {
            Log.d(TAG, "Starting LLM initialization...")

            if (!hasEnoughRam()) {
                Log.e(TAG, "Device does not have enough RAM for $MODEL_DISPLAY_NAME")
                return@withContext false
            }

            if (!isModelAvailable()) {
                Log.e(TAG, "Model asset not found: $MODEL_ASSET_PATH")
                return@withContext false
            }

            val modelFile = copyModelFromAssets()
            if (modelFile == null) {
                Log.e(TAG, "Failed to copy model from assets")
                return@withContext false
            }
            modelPath = modelFile.absolutePath
            Log.d(TAG, "Model file ready at: $modelPath (${modelFile.length() / 1024 / 1024} MB)")

            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath!!)
                .setMaxTokens(MAX_TOKENS)
                .setPreferredBackend(LlmInference.Backend.CPU)
                .build()

            Log.d(TAG, "Loading LLM model (this may take 10-30 seconds)...")
            llmInference = LlmInference.createFromOptions(context, options)

            isInitialized = true
            Log.d(TAG, "LLM initialized successfully with $MODEL_DISPLAY_NAME")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM", e)
            isInitialized = false
            false
        }
    }

    /**
     * Check if device has enough RAM.
     */
    private fun hasEnoughRam(): Boolean {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalRamMb = memInfo.totalMem / (1024 * 1024)
        val availableRamMb = memInfo.availMem / (1024 * 1024)
        Log.d(TAG, "Device RAM: ${totalRamMb}MB total, ${availableRamMb}MB available (need ${MIN_RAM_MB}MB)")

        return totalRamMb >= MIN_RAM_MB
    }

    /**
     * Check if the model asset exists.
     */
    private fun isModelAvailable(): Boolean {
        val localExists = File(context.filesDir, MODEL_FILE_NAME).exists()
        val assetExists = try {
            context.assets.open(MODEL_ASSET_PATH).close()
            true
        } catch (e: Exception) {
            false
        }

        Log.d(TAG, "Model $MODEL_DISPLAY_NAME: asset=$assetExists, local=$localExists")
        return assetExists || localExists
    }

    /**
     * Copy model from assets to internal storage.
     */
    private fun copyModelFromAssets(): File? {
        try {
            val modelFile = File(context.filesDir, MODEL_FILE_NAME)

            if (modelFile.exists()) {
                Log.d(TAG, "Model already exists at ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024} MB)")
                return modelFile
            }

            Log.d(TAG, "Copying $MODEL_DISPLAY_NAME from assets (this may take a minute)...")
            val startTime = System.currentTimeMillis()

            context.assets.open(MODEL_ASSET_PATH).use { inputStream ->
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
     * Generate a FULL response.
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
     * Generate a BRIEF 2-3 sentence summary — the primary RAG mode.
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

                Log.d(TAG, "Brief response in ${elapsed}ms: '${response.take(120)}'")

                val cleaned = cleanResponse(response)

                when {
                    cleaned.length < 5 -> {
                        Log.w(TAG, "Brief response too short (${cleaned.length} chars), extractive fallback")
                        buildExtractiveAnswer(query, contextChunks)
                    }
                    cleaned.length > 800 -> {
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
     * Streaming generation.
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
    // PROMPT ENGINEERING — Tuned for Gemma 2B
    // ============================================================================

    /**
     * Full RAG prompt — 2B can handle complex instructions reliably.
     */
    private fun buildRagPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(4).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(50)
            val section = if (chunk.sectionTitle.isNotBlank()) " — ${chunk.sectionTitle}" else ""
            val truncatedText = chunk.text.take(MAX_CHUNK_CHARS)
            "[Source ${index + 1}: $label$section, Page ${chunk.pageNumber}]\n$truncatedText"
        }.joinToString("\n---\n")

        val finalContext = if (contextText.length > MAX_CONTEXT_CHARS) {
            contextText.take(MAX_CONTEXT_CHARS)
        } else {
            contextText
        }

        return """<start_of_turn>user
You are a document research assistant. Your job is to answer questions using ONLY the provided source documents. Follow these rules strictly:

1. ONLY use information found in the sources below. Never use outside knowledge.
2. If the sources do not contain enough information to answer, say: "The loaded documents do not contain information about this topic."
3. Mention which source the information comes from.
4. Be concise and factual. No speculation or assumptions.
5. If multiple sources are relevant, synthesize information from all of them.

SOURCES:
$finalContext

QUESTION: $query
<end_of_turn>
<start_of_turn>model
"""
    }

    /**
     * Brief prompt — 2-3 sentence answer with source citation.
     */
    private fun buildBriefPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(3).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(50)
            val section = if (chunk.sectionTitle.isNotBlank()) " — ${chunk.sectionTitle}" else ""
            "[${label}${section}, p.${chunk.pageNumber}] ${chunk.text.take(400)}"
        }.joinToString("\n---\n")

        return """<start_of_turn>user
You are a document research assistant. Answer the question using ONLY the sources below.

Rules:
- Use ONLY facts from the sources. Do not add outside knowledge.
- If the answer is not in the sources, say "This is not covered in the loaded documents."
- Mention the source name in your answer.
- Keep your answer to 2-3 sentences maximum.

Sources:
$contextText

Question: $query
<end_of_turn>
<start_of_turn>model
"""
    }

    /**
     * Pure extractive fallback — no LLM needed.
     */
    private fun buildExtractiveAnswer(query: String, contextChunks: List<DocumentChunk>): String {
        if (contextChunks.isEmpty()) return "No relevant information found."

        val queryWords = query.lowercase().split(Regex("\\s+")).toSet()

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
            val topChunk = contextChunks.first()
            topChunk.text.take(200).trim() + "..."
        }
    }

    /**
     * Detect repetition loops.
     */
    private fun hasRepetitionLoop(text: String): Boolean {
        val words = text.split(Regex("\\s+"))
        if (words.size < 24) return false

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
     * Clean up the model's response.
     */
    private fun cleanResponse(response: String): String {
        return response
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .replace("<start_of_turn>", "")
            .replace("model\n", "")
            .replace(Regex("^\\s*Answer:\\s*"), "")
            .replace(Regex("^\\s*Based on the sources?:?\\s*"), "")
            .replace(Regex("^\\s*According to the sources?:?\\s*"), "")
            .trim()
    }

    fun isReady(): Boolean = isInitialized && llmInference != null

    fun isUpgradedModel(): Boolean = true

    fun getModelInfo(): String {
        return if (isInitialized) {
            MODEL_DISPLAY_NAME
        } else {
            "Loading..."
        }
    }

    fun close() {
        try {
            llmInference?.close()
            llmInference = null
            isInitialized = false
            Log.d(TAG, "LLM service closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing LLM service", e)
        }
    }
}