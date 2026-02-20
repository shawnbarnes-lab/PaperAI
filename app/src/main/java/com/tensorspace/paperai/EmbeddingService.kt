package com.tensorspace.paperai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.LongBuffer

/**
 * EmbeddingService generates text embeddings using the all-MiniLM-L6-v2 ONNX model.
 */
class EmbeddingService(private val context: Context) {

    companion object {
        private const val TAG = "EmbeddingService"
        private const val MODEL_PATH = "models/all-MiniLM-L6-v2.onnx"
        private const val TOKENIZER_PATH = "models/tokenizer.json"
        private const val EMBEDDING_DIM = 384
        private const val MAX_LENGTH = 256
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var vocabulary: Map<String, Int> = emptyMap()
    private var isInitialized = false

    /**
     * Initialize the embedding service.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) return@withContext true

        try {
            Log.d(TAG, "Initializing ONNX Runtime...")
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load the ONNX model
            Log.d(TAG, "Loading embedding model...")
            val modelBytes = context.assets.open(MODEL_PATH).readBytes()
            ortSession = ortEnvironment!!.createSession(modelBytes)
            Log.d(TAG, "Model loaded successfully")

            // Load tokenizer vocabulary
            Log.d(TAG, "Loading tokenizer...")
            loadTokenizer()
            Log.d(TAG, "Tokenizer loaded with ${vocabulary.size} tokens")

            isInitialized = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize embedding service", e)
            false
        }
    }

    /**
     * Load the tokenizer vocabulary from JSON.
     */
    private fun loadTokenizer() {
        val jsonString = context.assets.open(TOKENIZER_PATH).bufferedReader().readText()
        val json = JSONObject(jsonString)
        val model = json.getJSONObject("model")
        val vocab = model.getJSONObject("vocab")

        val vocabMap = mutableMapOf<String, Int>()
        vocab.keys().forEach { key ->
            vocabMap[key] = vocab.getInt(key)
        }
        vocabulary = vocabMap
    }

    /**
     * Generate an embedding for the given text.
     */
    suspend fun generateEmbedding(text: String): FloatArray = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            throw IllegalStateException("EmbeddingService not initialized")
        }

        try {
            // Tokenize the text
            val tokens = tokenize(text)

            // Create input tensors
            val inputIds = LongArray(tokens.size) { tokens[it].toLong() }
            val attentionMask = LongArray(tokens.size) { 1L }
            val tokenTypeIds = LongArray(tokens.size) { 0L }

            val env = ortEnvironment!!
            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(inputIds),
                longArrayOf(1, tokens.size.toLong())
            )
            val attentionMaskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attentionMask),
                longArrayOf(1, tokens.size.toLong())
            )
            val tokenTypeIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(tokenTypeIds),
                longArrayOf(1, tokens.size.toLong())
            )

            // Run inference
            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor,
                "token_type_ids" to tokenTypeIdsTensor
            )

            val results = ortSession!!.run(inputs)

            // Extract embeddings - get the [CLS] token embedding (first token)
            val outputTensor = results[0] as OnnxTensor
            val output = outputTensor.floatBuffer

            // The output shape is [1, seq_len, 384]
            // We want the mean of all token embeddings for sentence embedding
            val seqLen = tokens.size
            val embedding = FloatArray(EMBEDDING_DIM)

            // Mean pooling over all tokens
            for (i in 0 until seqLen) {
                for (j in 0 until EMBEDDING_DIM) {
                    embedding[j] += output.get(i * EMBEDDING_DIM + j)
                }
            }
            for (j in 0 until EMBEDDING_DIM) {
                embedding[j] /= seqLen
            }

            // Normalize the embedding
            val norm = Math.sqrt(embedding.sumOf { (it * it).toDouble() }).toFloat()
            for (i in embedding.indices) {
                embedding[i] /= norm
            }

            // Clean up
            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
            results.close()

            embedding

        } catch (e: Exception) {
            Log.e(TAG, "Error generating embedding", e)
            throw e
        }
    }

    /**
     * Simple tokenization using the vocabulary.
     */
    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        // Add [CLS] token
        val clsId = vocabulary["[CLS]"] ?: 101
        tokens.add(clsId)

        // Simple word-piece tokenization
        val words = text.lowercase().split(Regex("\\s+"))

        for (word in words) {
            if (tokens.size >= MAX_LENGTH - 1) break

            // Try to find the word in vocabulary
            val wordId = vocabulary[word]
            if (wordId != null) {
                tokens.add(wordId)
            } else {
                // Try subword tokenization
                var remaining = word
                var isFirst = true

                while (remaining.isNotEmpty() && tokens.size < MAX_LENGTH - 1) {
                    var found = false

                    // Try progressively shorter prefixes
                    for (endIdx in remaining.length downTo 1) {
                        val subword = if (isFirst) {
                            remaining.substring(0, endIdx)
                        } else {
                            "##" + remaining.substring(0, endIdx)
                        }

                        val subwordId = vocabulary[subword]
                        if (subwordId != null) {
                            tokens.add(subwordId)
                            remaining = remaining.substring(endIdx)
                            isFirst = false
                            found = true
                            break
                        }
                    }

                    if (!found) {
                        // Use [UNK] token for unknown characters
                        val unkId = vocabulary["[UNK]"] ?: 100
                        tokens.add(unkId)
                        remaining = remaining.drop(1)
                        isFirst = false
                    }
                }
            }
        }

        // Add [SEP] token
        val sepId = vocabulary["[SEP]"] ?: 102
        tokens.add(sepId)

        return tokens
    }

    /**
     * Close the service and release resources.
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        isInitialized = false
        Log.d(TAG, "EmbeddingService closed")
    }
}