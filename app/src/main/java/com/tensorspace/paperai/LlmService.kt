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
 * v4.0 — Gemma 2B Q8 — Tuned for fuller, refined responses
 * - Increased token limits for complete answers
 * - Refined prompts for better instruction following
 * - Better source attribution
 * - Improved response cleaning
 */
class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MAX_TOKENS = 2048          // ← Doubled from 1024
        private const val MAX_CONTEXT_CHARS = 3000    // ← Increased from 2000
        private const val MAX_CHUNK_CHARS = 600       // ← Increased from 500
        private const val MODEL_ASSET_PATH = "models/gemma-2b-int4.task"
        private const val MODEL_FILE_NAME = "gemma-2b-int4.task"
        private const val MODEL_DISPLAY_NAME = "Gemma 2B (Q8)"
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
                  // ← Low temp = more focused/factual
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

    private fun hasEnoughRam(): Boolean {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalRamMb = memInfo.totalMem / (1024 * 1024)
        val availableRamMb = memInfo.availMem / (1024 * 1024)
        Log.d(TAG, "Device RAM: ${totalRamMb}MB total, ${availableRamMb}MB available (need ${MIN_RAM_MB}MB)")

        return totalRamMb >= MIN_RAM_MB
    }

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
     * Generate a FULL detailed response.
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

                Log.d(TAG, "Response generated in ${elapsed}ms (${response.length} chars)")
                cleanResponse(response)

            } catch (e: Exception) {
                Log.e(TAG, "Error generating response: ${e.javaClass.simpleName}: ${e.message}", e)
                "I encountered an error while generating a response. Please try again."
            }
        }

    /**
     * Generate a focused 3-5 sentence summary — the primary RAG mode.
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

                Log.d(TAG, "Brief response in ${elapsed}ms (${response.length} chars)")

                val cleaned = cleanResponse(response)

                when {
                    cleaned.length < 10 -> {
                        Log.w(TAG, "Brief response too short (${cleaned.length} chars), extractive fallback")
                        buildExtractiveAnswer(query, contextChunks)
                    }
                    cleaned.length > 1200 -> {
                        Log.w(TAG, "Brief response too long (${cleaned.length} chars), trimming")
                        trimToCompleteSentences(cleaned, 1000)
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
    // PROMPT ENGINEERING — Tuned for Gemma 2B Q8 (fuller, more refined responses)
    // ============================================================================

    /**
     * Full RAG prompt — detailed, comprehensive answer with source citations.
     */
    private fun buildRagPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(5).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(60)
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
You are a precise document research assistant. Answer the question thoroughly using ONLY the provided sources.

Instructions:
1. Use ONLY information from the sources below. Never add outside knowledge.
2. If the sources don't contain the answer, say: "The loaded documents do not contain information about this topic."
3. Reference which source the information comes from (e.g., "According to Source 1...").
4. Provide a complete, well-structured answer. Include relevant details, examples, and explanations found in the sources.
5. If multiple sources contain relevant information, synthesize them into a coherent answer.
6. Use direct quotes from the sources when they are particularly relevant.

SOURCES:
$finalContext

QUESTION: $query

Provide a thorough, well-organized answer with source citations:
<end_of_turn>
<start_of_turn>model
"""
    }

    /**
     * Brief prompt — focused 3-5 sentence answer with source citation.
     */
    private fun buildBriefPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(4).mapIndexed { index, chunk ->
            val label = chunk.sourceName.take(60)
            val section = if (chunk.sectionTitle.isNotBlank()) " — ${chunk.sectionTitle}" else ""
            "[Source ${index + 1}: ${label}${section}, p.${chunk.pageNumber}]\n${chunk.text.take(500)}"
        }.joinToString("\n---\n")

        return """<start_of_turn>user
You are a document research assistant. Answer the question using ONLY the sources below.

Rules:
- Use ONLY facts from the sources. Do not add outside knowledge.
- If the answer is not in the sources, say "This is not covered in the loaded documents."
- Mention which source the information comes from.
- Give a focused answer in 3-5 complete sentences.
- Include specific details, numbers, or key terms from the sources.

Sources:
$contextText

Question: $query

Answer in 3-5 sentences with source references:
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

        val scoredSentences = mutableListOf<Triple<String, Int, String>>()

        for (chunk in contextChunks.take(5)) {
            val sourceName = chunk.sourceName
                .replace(".pdf", "")
                .replace("_", " ")

            val sentences = chunk.text
                .split(Regex("[.!?]+\\s+"))
                .filter { it.length > 20 }

            for (sentence in sentences) {
                val sentenceWords = sentence.lowercase().split(Regex("\\s+")).toSet()
                val score = queryWords.intersect(sentenceWords).size
                scoredSentences.add(Triple(sentence.trim(), score, sourceName))
            }
        }

        // Get top 2-3 most relevant sentences
        val topSentences = scoredSentences
            .sortedByDescending { it.second }
            .take(3)
            .filter { it.second > 0 }

        return if (topSentences.isNotEmpty()) {
            val mainSource = topSentences.first().third
            val combined = topSentences.joinToString(". ") { it.first }
            "From $mainSource: $combined."
        } else {
            val topChunk = contextChunks.first()
            topChunk.text.take(300).trim() + "..."
        }
    }

    /**
     * Trim text to complete sentences within a character limit.
     */
    private fun trimToCompleteSentences(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text

        val truncated = text.take(maxChars)
        val lastPeriod = truncated.lastIndexOf('.')
        val lastExclamation = truncated.lastIndexOf('!')
        val lastQuestion = truncated.lastIndexOf('?')

        val lastSentenceEnd = maxOf(lastPeriod, lastExclamation, lastQuestion)

        return if (lastSentenceEnd > maxChars / 2) {
            truncated.take(lastSentenceEnd + 1).trim()
        } else {
            truncated.trim() + "..."
        }
    }

    /**
     * Detect repetition loops.
     */
    private fun hasRepetitionLoop(text: String): Boolean {
        val words = text.split(Regex("\\s+"))
        if (words.size < 24) return false

        val windowSize = 6
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
        var cleaned = response
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .replace("<start_of_turn>", "")
            .replace("<bos>", "")
            .replace("model\n", "")
            .replace(Regex("^\\s*Answer:\\s*", RegexOption.IGNORE_CASE), "")
            .replace(Regex("^\\s*Response:\\s*", RegexOption.IGNORE_CASE), "")
            .trim()

        // Remove any trailing incomplete sentence
        if (cleaned.isNotEmpty() && !cleaned.last().let { it == '.' || it == '!' || it == '?' || it == '"' }) {
            val lastSentenceEnd = maxOf(
                cleaned.lastIndexOf('.'),
                cleaned.lastIndexOf('!'),
                cleaned.lastIndexOf('?')
            )
            if (lastSentenceEnd > cleaned.length / 2) {
                cleaned = cleaned.take(lastSentenceEnd + 1).trim()
            }
        }

        return cleaned
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