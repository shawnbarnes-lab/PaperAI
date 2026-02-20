package com.tensorspace.paperai

import android.app.Application
import android.util.Log

/**
 * Custom Application class for paperAI.
 *
 * The Application class is the first thing Android creates when your app starts.
 * It exists for the entire lifetime of your app (until Android kills it).
 * This makes it perfect for one-time initialization tasks like setting up databases.
 *
 * To use a custom Application class, we must register it in AndroidManifest.xml
 * (we'll do that in the next step).
 */
class PaperAIApplication : Application() {

    companion object {
        // Tag for logging - helps filter log messages in Logcat
        private const val TAG = "PaperAIApplication"
    }

    /**
     * Called when the application is starting, before any activity, service,
     * or receiver objects have been created.
     *
     * This is where we initialize our database. By doing it here, we guarantee
     * the database is ready before any screen tries to use it.
     */
    override fun onCreate() {
        super.onCreate()

        Log.d(TAG, "Application starting - initializing ObjectBox database...")

        // Initialize the ObjectBox database
        // This creates the database files if they don't exist, or opens
        // existing ones if the app has been run before.
        ObjectBox.init(this)

        Log.d(TAG, "ObjectBox initialized successfully!")
    }
}