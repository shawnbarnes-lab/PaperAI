package com.tensorspace.paperai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.tensorspace.paperai.ui.theme.PaperAITheme

/**
 * MainActivity is the entry point for the paperAI app.
 *
 * ANDROID ACTIVITY BASICS:
 * ------------------------
 * An Activity represents a single screen in an Android app. When you tap the
 * paperAI icon on your phone, Android creates this MainActivity and calls
 * its onCreate() method.
 *
 * In traditional Android development, Activities managed their own UI using
 * XML layouts. With Jetpack Compose, the Activity's main job is simply to
 * host the Compose UI by calling setContent{}.
 *
 * WHY ComponentActivity:
 * ----------------------
 * We extend ComponentActivity (not the older AppCompatActivity) because it's
 * the base class designed for Jetpack Compose. It provides the setContent{}
 * function and integrates well with Compose's lifecycle.
 *
 * THE FLOW:
 * ---------
 * 1. User taps app icon
 * 2. Android creates MainActivity
 * 3. onCreate() is called
 * 4. setContent{} installs our Compose UI (MainScreen)
 * 5. MainScreen creates/accesses MainViewModel
 * 6. MainViewModel initializes RagEngine
 * 7. User sees the app and can start searching!
 */
class MainActivity : ComponentActivity() {

    /**
     * Called when the activity is first created.
     *
     * This is where we set up our UI. The savedInstanceState parameter contains
     * data saved from a previous instance (if the activity was destroyed and
     * recreated, e.g., during rotation), but Compose handles most state
     * preservation automatically through the ViewModel.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable edge-to-edge display (content extends behind system bars)
        // This gives a more modern, immersive look on newer Android versions
        enableEdgeToEdge()

        // Set the Compose content
        // Everything inside setContent{} is Compose UI code
        setContent {
            // Apply the app's theme (colors, typography, shapes)
            // PaperAITheme was auto-generated in ui.theme when you created the project
            PaperAITheme {
                // Surface is a basic container that applies the theme's background color
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // MainScreen is our app's main UI
                    // It handles everything: search bar, tabs, results display
                    MainScreen()
                }
            }
        }
    }
}