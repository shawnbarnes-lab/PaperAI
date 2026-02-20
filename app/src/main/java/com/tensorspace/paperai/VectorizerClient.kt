package com.tensorspace.paperai

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * VectorizerClient handles communication with the remote vectorizer server.
 * 
 * Sends documents to the server for GPU-accelerated embedding generation,
 * then receives the vectorized result for local import.
 */
class VectorizerClient(private val context: Context) {
    
    companion object {
        private const val TAG = "VectorizerClient"
        private const val TIMEOUT_MS = 300_000 // 5 minutes for large docs
        private const val BOUNDARY = "----PaperAIBoundary"
    }
    
    /**
     * Result of a vectorization request.
     */
    data class VectorizeResult(
        val success: Boolean,
        val vectorJson: String? = null,
        val chunkCount: Int = 0,
        val error: String? = null
    )
    
    /**
     * Server health status.
     */
    data class ServerStatus(
        val online: Boolean,
        val device: String? = null,
        val gpu: String? = null,
        val error: String? = null
    )
    
    /**
     * Check if the server is online and get its status.
     */
    suspend fun checkServer(serverUrl: String): ServerStatus = withContext(Dispatchers.IO) {
        try {
            val url = URL("$serverUrl/health")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            
            val responseCode = connection.responseCode
            
            if (responseCode == 200) {
                val response = connection.inputStream.bufferedReader().readText()
                val json = JSONObject(response)
                
                ServerStatus(
                    online = true,
                    device = json.optString("device"),
                    gpu = json.optString("gpu")
                )
            } else {
                ServerStatus(
                    online = false,
                    error = "Server returned: $responseCode"
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Server check failed", e)
            ServerStatus(
                online = false,
                error = e.message ?: "Connection failed"
            )
        }
    }
    
    /**
     * Send a document to the server for vectorization.
     * 
     * @param serverUrl Base URL of the vectorizer server (e.g., "http://192.168.1.100:5000")
     * @param documentUri URI of the document to vectorize
     * @param onProgress Optional progress callback (0.0 to 1.0)
     */
    suspend fun vectorizeDocument(
        serverUrl: String,
        documentUri: Uri,
        onProgress: ((Float, String) -> Unit)? = null
    ): VectorizeResult = withContext(Dispatchers.IO) {
        try {
            onProgress?.invoke(0.1f, "Preparing document...")
            
            // Get filename from URI
            val filename = getFilename(documentUri) ?: "document.txt"
            Log.d(TAG, "Vectorizing: $filename")
            
            // Read file content
            val fileBytes = context.contentResolver.openInputStream(documentUri)?.use { 
                it.readBytes() 
            } ?: return@withContext VectorizeResult(
                success = false,
                error = "Could not read file"
            )
            
            Log.d(TAG, "File size: ${fileBytes.size} bytes")
            onProgress?.invoke(0.2f, "Uploading to server...")
            
            // Build multipart request
            val url = URL("$serverUrl/vectorize")
            val connection = url.openConnection() as HttpURLConnection
            
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = TIMEOUT_MS
            connection.readTimeout = TIMEOUT_MS
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$BOUNDARY")
            
            // Write multipart body
            DataOutputStream(connection.outputStream).use { dos ->
                // File part
                dos.writeBytes("--$BOUNDARY\r\n")
                dos.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"$filename\"\r\n")
                dos.writeBytes("Content-Type: application/octet-stream\r\n")
                dos.writeBytes("\r\n")
                dos.write(fileBytes)
                dos.writeBytes("\r\n")
                
                // End boundary
                dos.writeBytes("--$BOUNDARY--\r\n")
                dos.flush()
            }
            
            onProgress?.invoke(0.5f, "Processing on server...")
            
            // Read response
            val responseCode = connection.responseCode
            Log.d(TAG, "Server response code: $responseCode")
            
            if (responseCode == 200) {
                onProgress?.invoke(0.9f, "Receiving vectors...")
                
                val response = connection.inputStream.bufferedReader().readText()
                val json = JSONObject(response)
                val chunkCount = json.optInt("chunkCount", 0)
                
                Log.d(TAG, "Received $chunkCount chunks")
                onProgress?.invoke(1.0f, "Complete!")
                
                VectorizeResult(
                    success = true,
                    vectorJson = response,
                    chunkCount = chunkCount
                )
            } else {
                val errorStream = connection.errorStream?.bufferedReader()?.readText()
                val errorJson = try {
                    JSONObject(errorStream ?: "{}").optString("error", "Unknown error")
                } catch (e: Exception) {
                    errorStream ?: "Server error: $responseCode"
                }
                
                Log.e(TAG, "Server error: $errorJson")
                VectorizeResult(
                    success = false,
                    error = errorJson
                )
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Vectorization failed", e)
            VectorizeResult(
                success = false,
                error = e.message ?: "Network error"
            )
        }
    }
    
    /**
     * Get filename from URI.
     */
    private fun getFilename(uri: Uri): String? {
        // Try to get display name from content resolver
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    return cursor.getString(nameIndex)
                }
            }
        }
        
        // Fall back to last path segment
        return uri.lastPathSegment
    }
}
