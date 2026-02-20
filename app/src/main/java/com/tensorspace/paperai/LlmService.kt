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
 * This service loads the Gemma 3 1B model and generates responses based on
 * retrieved context chunks for RAG (Retrieval-Augmented Generation).
 */
class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MODEL_ASSET_NAME = "models/gemma3.task"
        private const val MODEL_FILE_NAME = "gemma3.task"
        private const val MAX_TOKENS = 1024
        private const val TEMPERATURE = 0.7f
        private const val TOP_K = 40
        private const val TOP_P = 0.95f
    }

    private var llmInference: LlmInference? = null
    private var isInitialized = false
    private var modelPath: String? = null

    /**
     * Initialize the LLM service by copying the model from assets and loading it.
     * This should be called once when the app starts.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) {
            Log.d(TAG, "LLM already initialized")
            return@withContext true
        }

        try {
            Log.d(TAG, "Starting LLM initialization...")

            // Step 1: Copy model from assets to internal storage (if needed)
            val modelFile = copyModelFromAssets()
            if (modelFile == null) {
                Log.e(TAG, "Failed to copy model from assets")
                return@withContext false
            }
            modelPath = modelFile.absolutePath
            Log.d(TAG, "Model file ready at: $modelPath")

            // Step 2: Create LLM Inference options
            // Must set PreferredBackend explicitly for newer MediaPipe versions
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath!!)
                .setMaxTokens(MAX_TOKENS)
                .setPreferredBackend(LlmInference.Backend.CPU)
                .build()

            // Step 3: Create the LLM Inference instance
            Log.d(TAG, "Loading LLM model (this may take 10-30 seconds)...")
            llmInference = LlmInference.createFromOptions(context, options)

            isInitialized = true
            Log.d(TAG, "LLM initialized successfully!")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM", e)
            isInitialized = false
            false
        }
    }

    /**
     * Copy the model file from assets to internal storage.
     * MediaPipe requires a file path, not an asset stream.
     */
    private fun copyModelFromAssets(): File? {
        try {
            val modelFile = File(context.filesDir, MODEL_FILE_NAME)

            // Check if already copied
            if (modelFile.exists()) {
                Log.d(TAG, "Model already exists at ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024} MB)")
                return modelFile
            }

            Log.d(TAG, "Copying model from assets (this may take a minute)...")
            val startTime = System.currentTimeMillis()

            context.assets.open(MODEL_ASSET_NAME).use { inputStream ->
                FileOutputStream(modelFile).use { outputStream ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    var totalBytes = 0L

                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                        totalBytes += bytesRead

                        // Log progress every 50MB
                        if (totalBytes % (50 * 1024 * 1024) < 8192) {
                            Log.d(TAG, "Copied ${totalBytes / 1024 / 1024} MB...")
                        }
                    }
                }
            }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Model copied successfully in ${elapsed}ms (${modelFile.length() / 1024 / 1024} MB)")
            return modelFile

        } catch (e: Exception) {
            Log.e(TAG, "Error copying model from assets", e)
            return null
        }
    }

    /**
     * Generate a response based on the query and retrieved context chunks.
     * This is the main RAG generation method.
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
                // Build the RAG prompt
                val prompt = buildRagPrompt(query, contextChunks)
                Log.d(TAG, "Generating response for query: '$query'")
                Log.d(TAG, "Prompt length: ${prompt.length} characters, ~${prompt.length / 4} tokens")
                Log.d(TAG, "First 200 chars of prompt: ${prompt.take(200)}")

                // Generate response
                val startTime = System.currentTimeMillis()
                val response = llmInference!!.generateResponse(prompt)
                val elapsed = System.currentTimeMillis() - startTime

                Log.d(TAG, "Response generated in ${elapsed}ms")
                
                // Clean up the response
                cleanResponse(response)

            } catch (e: Exception) {
                Log.e(TAG, "Error generating response: ${e.javaClass.simpleName}: ${e.message}", e)
                "I encountered an error while generating a response. Please try again. (${e.javaClass.simpleName})"
            }
        }

    /**
     * Generate a response with streaming (token by token).
     * Calls onToken for each generated token.
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
            val msg = "I couldn't find any relevant information in the documents to answer your question."
            onToken(msg)
            return@withContext msg
        }

        try {
            val prompt = buildRagPrompt(query, contextChunks)
            Log.d(TAG, "Starting streaming generation...")

            // Use synchronous generation and simulate streaming
            // (MediaPipe's async streaming is complex to set up properly)
            val response = llmInference!!.generateResponse(prompt)
            val cleanedResponse = cleanResponse(response)

            // Stream the response word by word for UI feedback
            val words = cleanedResponse.split(" ")
            val result = StringBuilder()

            for (word in words) {
                val token = if (result.isEmpty()) word else " $word"
                result.append(token)
                onToken(token)
                // Small delay for visual streaming effect
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

    /**
     * Build the RAG prompt with context and query.
     * Uses Gemma's chat format.
     */
    private fun buildRagPrompt(query: String, contextChunks: List<DocumentChunk>): String {
        val contextText = contextChunks.take(3).mapIndexed { index, chunk ->
            "[${index + 1}] ${chunk.sourceName}: ${chunk.text}"
        }.joinToString("\n\n")

        // Limit context to prevent exceeding model's context window
        val maxContextLength = 1500
        val truncatedContext = if (contextText.length > maxContextLength) {
            contextText.take(maxContextLength)
        } else {
            contextText
        }
        
        // Simplified prompt for small model - very direct instructions
        return """<start_of_turn>user
Read the sources below, then answer the question.

SOURCES:
$truncatedContext

QUESTION: $query

Give a direct, accurate answer using only the information from the sources above. Reference which source the information comes from.
<end_of_turn>
<start_of_turn>model
"""
    }

    /**
     * Clean up the model's response by removing any artifacts.
     */
    private fun cleanResponse(response: String): String {
        return response
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .replace("<start_of_turn>", "")
            .replace("model\n", "")
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
        return if (isInitialized) {
            "Gemma 3 1B (int4 quantized)"
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
            Log.d(TAG, "LLM service closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing LLM service", e)
        }
    }
}
