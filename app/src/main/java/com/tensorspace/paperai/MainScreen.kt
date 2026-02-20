package com.tensorspace.paperai

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * MainScreen is the primary UI for paperAI.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val searchQuery by viewModel.searchQuery.collectAsState()
    val selectedTab by viewModel.selectedTab.collectAsState()
    val importResult by viewModel.importResult.collectAsState()
    val serverUrl by viewModel.serverUrl.collectAsState()
    val uploadProgress by viewModel.uploadProgress.collectAsState()
    val serverStatus by viewModel.serverStatus.collectAsState()
    
    // State for document viewer dialog
    var selectedDocument by remember { mutableStateOf<String?>(null) }
    var documentChunks by remember { mutableStateOf<List<DocumentChunk>>(emptyList()) }
    var highlightedChunkId by remember { mutableStateOf<Long?>(null) }
    
    // State for server settings dialog
    var showServerDialog by remember { mutableStateOf(false) }
    
    // File picker for importing .paperai files (local import)
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.importVectors(it) }
    }
    
    // File picker for uploading documents to server
    val uploadPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.uploadAndVectorize(it) }
    }

    // Show document viewer dialog when a document is selected
    if (selectedDocument != null) {
        DocumentViewerDialog(
            documentName = selectedDocument!!,
            chunks = documentChunks,
            highlightedChunkId = highlightedChunkId,
            onDismiss = { 
                selectedDocument = null
                documentChunks = emptyList()
                highlightedChunkId = null
            }
        )
    }
    
    // Show server settings dialog
    if (showServerDialog) {
        ServerSettingsDialog(
            currentUrl = serverUrl,
            serverStatus = serverStatus,
            onUrlChange = { viewModel.setServerUrl(it) },
            onCheckServer = { viewModel.checkServer() },
            onDismiss = { showServerDialog = false }
        )
    }
    
    // Show upload progress dialog
    uploadProgress?.let { progress ->
        UploadProgressDialog(
            progress = progress,
            onCancel = { viewModel.cancelUpload() }
        )
    }
    
    // Show import result snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    LaunchedEffect(importResult) {
        importResult?.let { result ->
            val message = if (result.success) {
                "Document added! ${result.chunksImported} sections ready to search"
            } else {
                "Couldn't add document: ${result.error}"
            }
            snackbarHostState.showSnackbar(message)
            viewModel.clearImportResult()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "📚",
                            style = MaterialTheme.typography.titleLarge
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "AI Knowledge Base",
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    // Upload to server button
                    IconButton(
                        onClick = { 
                            uploadPickerLauncher.launch(arrayOf("text/plain", "application/pdf", "*/*"))
                        },
                        enabled = uiState.isEngineReady && !uiState.isLoading && uploadProgress == null
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "Add Document"
                        )
                    }
                    // Server settings button
                    IconButton(
                        onClick = { showServerDialog = true }
                    ) {
                        Icon(
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            StatusBar(uiState = uiState)

            SearchBar(
                query = searchQuery,
                onQueryChange = viewModel::onSearchQueryChanged,
                onSearch = viewModel::onSearch,
                onClear = viewModel::onClearSearch,
                enabled = uiState.canSearch()
            )

            if (uiState.isInitializing) {
                InitializingView()
            } else if (uiState.lastResponse != null || uiState.isSearching) {
                TabSection(
                    selectedTab = selectedTab,
                    onTabSelected = viewModel::onTabSelected,
                    uiState = uiState,
                    onSourceClick = { chunk ->
                        // Load all chunks for this document and show dialog
                        // Pass the clicked chunk's ID so we can highlight it
                        selectedDocument = chunk.sourceName
                        documentChunks = viewModel.getDocumentChunks(chunk.sourceName)
                        highlightedChunkId = chunk.id
                    }
                )
            } else {
                WelcomeView(
                    hasDocuments = uiState.stats?.hasDocuments() == true,
                    onLoadSamples = viewModel::loadSampleDocuments,
                    onImport = { filePickerLauncher.launch(arrayOf("application/json", "*/*")) },
                    onUpload = { uploadPickerLauncher.launch(arrayOf("text/plain", "application/pdf", "*/*")) },
                    onServerSettings = { showServerDialog = true }
                )
            }
        }
    }
}

@Composable
fun StatusBar(uiState: UiState) {
    val backgroundColor = when {
        uiState.error != null -> MaterialTheme.colorScheme.errorContainer
        uiState.isInitializing -> MaterialTheme.colorScheme.secondaryContainer
        uiState.isEngineReady -> MaterialTheme.colorScheme.tertiaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }

    val textColor = when {
        uiState.error != null -> MaterialTheme.colorScheme.onErrorContainer
        uiState.isInitializing -> MaterialTheme.colorScheme.onSecondaryContainer
        uiState.isEngineReady -> MaterialTheme.colorScheme.onTertiaryContainer
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = backgroundColor
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (uiState.shouldShowLoading()) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = textColor
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(
                text = uiState.getStatusMessage(),
                style = MaterialTheme.typography.bodySmall,
                color = textColor
            )
        }
    }
}

@Composable
fun SearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    onSearch: () -> Unit,
    onClear: () -> Unit,
    enabled: Boolean
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        shape = RoundedCornerShape(28.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 2.dp
    ) {
        TextField(
            value = query,
            onValueChange = onQueryChange,
            modifier = Modifier.fillMaxWidth(),
            enabled = enabled,
            placeholder = {
                Text("Ask anything...")
            },
            leadingIcon = {
                Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = "Search"
                )
            },
            trailingIcon = {
                Row {
                    if (query.isNotEmpty()) {
                        IconButton(onClick = onClear) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = "Clear"
                            )
                        }
                    }
                    IconButton(
                        onClick = onSearch,
                        enabled = enabled && query.isNotBlank()
                    ) {
                        Icon(
                            imageVector = Icons.Default.Search,
                            contentDescription = "Execute Search",
                            tint = if (enabled && query.isNotBlank())
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            },
            keyboardOptions = KeyboardOptions(
                imeAction = ImeAction.Search
            ),
            keyboardActions = KeyboardActions(
                onSearch = { onSearch() }
            ),
            singleLine = true,
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                disabledContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent
            )
        )
    }
}

@Composable
fun TabSection(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit,
    uiState: UiState,
    onSourceClick: (DocumentChunk) -> Unit = {}
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = MaterialTheme.colorScheme.surface,
            contentColor = MaterialTheme.colorScheme.primary
        ) {
            Tab(
                selected = selectedTab == 0,
                onClick = { onTabSelected(0) },
                text = { Text("AI Response") }
            )
            Tab(
                selected = selectedTab == 1,
                onClick = { onTabSelected(1) },
                text = {
                    val count = uiState.lastResponse?.sources?.size ?: 0
                    Text("Sources ($count)")
                }
            )
        }

        when (selectedTab) {
            0 -> ResponseTab(uiState = uiState)
            1 -> SourcesTab(uiState = uiState, onSourceClick = onSourceClick)
        }
    }
}

@Composable
fun ResponseTab(uiState: UiState) {
    val response = uiState.lastResponse

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        if (uiState.isSearching) {
            Column(
                modifier = Modifier.align(Alignment.Center),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Generating response...")
            }
        } else if (response != null) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = "You asked: \"${response.query}\"",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(12.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "AI Response",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = response.response,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "Processed in ${formatProcessingTime(response.processingTimeMs)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

/**
 * Format processing time for display.
 */
fun formatProcessingTime(timeMs: Long): String {
    return when {
        timeMs < 1000 -> "${timeMs}ms"
        else -> String.format("%.1fs", timeMs / 1000.0)
    }
}

@Composable
fun SourcesTab(
    uiState: UiState,
    onSourceClick: (DocumentChunk) -> Unit = {}
) {
    val sources = uiState.lastResponse?.sources ?: emptyList()

    if (uiState.isSearching) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Searching documents...")
            }
        }
    } else if (sources.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "No sources found",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    } else {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(sources) { chunk ->
                SourceCard(
                    chunk = chunk,
                    onClick = { onSourceClick(chunk) }
                )
            }
        }
    }
}

@Composable
fun SourceCard(
    chunk: DocumentChunk,
    onClick: () -> Unit = {}
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = chunk.sourceName,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Page ${chunk.pageNumber}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = getRelevanceColor(chunk.similarity)
                ) {
                    Text(
                        text = "${(chunk.similarity * 100).toInt()}%",
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = chunk.text,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = getRelevanceLabel(chunk.similarity),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontStyle = FontStyle.Italic
                )
                Text(
                    text = "Tap to view full document",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    fontStyle = FontStyle.Italic
                )
            }
        }
    }
}

/**
 * Full-screen dialog to view an entire document.
 * Highlights and auto-scrolls to the matched chunk.
 */
@Composable
fun DocumentViewerDialog(
    documentName: String,
    chunks: List<DocumentChunk>,
    highlightedChunkId: Long? = null,
    onDismiss: () -> Unit
) {
    val sortedChunks = remember(chunks) { chunks.sortedBy { it.pageNumber } }
    val listState = rememberLazyListState()
    
    // Auto-scroll to highlighted chunk when dialog opens
    LaunchedEffect(highlightedChunkId, sortedChunks) {
        if (highlightedChunkId != null) {
            val index = sortedChunks.indexOfFirst { it.id == highlightedChunkId }
            if (index >= 0) {
                listState.animateScrollToItem(index)
            }
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false
        )
    ) {
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surface
        ) {
            Column(
                modifier = Modifier.fillMaxSize()
            ) {
                // Header
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = documentName,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Text(
                                text = "${sortedChunks.size} sections",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                        IconButton(onClick = onDismiss) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Close",
                                tint = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                    }
                }

                // Document content
                if (sortedChunks.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "No content available",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        items(sortedChunks) { chunk ->
                            DocumentSection(
                                chunk = chunk,
                                isHighlighted = chunk.id == highlightedChunkId
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DocumentSection(
    chunk: DocumentChunk,
    isHighlighted: Boolean = false
) {
    val backgroundColor = if (isHighlighted) {
        MaterialTheme.colorScheme.tertiaryContainer
    } else {
        Color.Transparent
    }
    
    val borderColor = if (isHighlighted) {
        MaterialTheme.colorScheme.tertiary
    } else {
        Color.Transparent
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = backgroundColor,
        border = if (isHighlighted) {
            BorderStroke(2.dp, borderColor)
        } else null
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Page ${chunk.pageNumber}",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = if (isHighlighted) {
                        MaterialTheme.colorScheme.onTertiaryContainer
                    } else {
                        MaterialTheme.colorScheme.primary
                    }
                )
                if (isHighlighted) {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.tertiary
                    ) {
                        Text(
                            text = "MATCHED",
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onTertiary
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = chunk.text,
                style = MaterialTheme.typography.bodyMedium,
                color = if (isHighlighted) {
                    MaterialTheme.colorScheme.onTertiaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurface
                }
            )
            if (!isHighlighted) {
                Spacer(modifier = Modifier.height(8.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            }
        }
    }
}

@Composable
fun getRelevanceColor(score: Float): Color {
    return when {
        score >= 0.8f -> Color(0xFF2E7D32)
        score >= 0.6f -> Color(0xFF558B2F)
        score >= 0.4f -> Color(0xFFF9A825)
        score >= 0.2f -> Color(0xFFEF6C00)
        else -> Color(0xFFD32F2F)
    }
}

fun getRelevanceLabel(score: Float): String {
    return when {
        score >= 0.8f -> "Highly relevant"
        score >= 0.6f -> "Relevant"
        score >= 0.4f -> "Somewhat relevant"
        score >= 0.2f -> "Marginally relevant"
        else -> "Low relevance"
    }
}

@Composable
fun InitializingView() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(64.dp),
                strokeWidth = 4.dp
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "Loading AI Models...",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "This may take a moment on first launch",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun WelcomeView(
    hasDocuments: Boolean,
    onLoadSamples: () -> Unit,
    onImport: () -> Unit = {},
    onUpload: () -> Unit = {},
    onServerSettings: () -> Unit = {}
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .padding(32.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // App icon/logo placeholder
            Surface(
                modifier = Modifier.size(80.dp),
                shape = RoundedCornerShape(20.dp),
                color = MaterialTheme.colorScheme.primaryContainer
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text(
                        text = "📚",
                        style = MaterialTheme.typography.displaySmall
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            Text(
                text = "Your Knowledge,\nAlways Offline",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = if (hasDocuments) {
                    "Ask anything about your documents"
                } else {
                    "Turn any document into a searchable AI assistant that works without internet"
                },
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )

            if (!hasDocuments) {
                Spacer(modifier = Modifier.height(32.dp))
                
                // How it works section
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        horizontalAlignment = Alignment.Start
                    ) {
                        Text(
                            text = "How it works",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        HowItWorksStep(
                            number = "1",
                            text = "Add your document (PDF)"
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        HowItWorksStep(
                            number = "2", 
                            text = "We process it into a knowledge pack"
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        HowItWorksStep(
                            number = "3",
                            text = "Ask questions anytime - no internet needed"
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Primary action - Add document
                Button(
                    onClick = onUpload,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = PaddingValues(vertical = 16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Add Your First Document",
                        style = MaterialTheme.typography.titleSmall
                    )
                }
                
                Spacer(modifier = Modifier.height(24.dp))
                
                HorizontalDivider(
                    modifier = Modifier.fillMaxWidth(0.6f),
                    color = MaterialTheme.colorScheme.outlineVariant
                )
                
                Spacer(modifier = Modifier.height(20.dp))
                
                Text(
                    text = "Or get started quickly",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Try sample content
                    OutlinedButton(
                        onClick = onLoadSamples,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text("Try Sample Content")
                    }

                    // Import existing pack
                    OutlinedButton(
                        onClick = onImport,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text("Import Pack")
                    }
                }
                
                Spacer(modifier = Modifier.height(20.dp))
                
                // What's a pack? Help text
                Text(
                    text = "Already have a .pack file? Tap 'Import Pack' to load it instantly.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Server settings (subtle)
                TextButton(onClick = onServerSettings) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Processing Settings",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun HowItWorksStep(number: String, text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Surface(
            modifier = Modifier.size(24.dp),
            shape = RoundedCornerShape(6.dp),
            color = MaterialTheme.colorScheme.primary
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = number,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimary
                )
            }
        }
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

/**
 * Dialog for configuring the processing server URL.
 */
@Composable
fun ServerSettingsDialog(
    currentUrl: String,
    serverStatus: VectorizerClient.ServerStatus?,
    onUrlChange: (String) -> Unit,
    onCheckServer: () -> Unit,
    onDismiss: () -> Unit
) {
    var urlText by remember { mutableStateOf(currentUrl) }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Processing Server") },
        text = {
            Column {
                Text(
                    text = "Documents are processed on a remote server for faster performance. Enter your server address below.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                OutlinedTextField(
                    value = urlText,
                    onValueChange = { urlText = it },
                    label = { Text("Server Address") },
                    placeholder = { Text("https://your-server.ngrok-free.app") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Server status
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    serverStatus?.let { status ->
                        if (status.online) {
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = Color(0xFF22C55E)
                            ) {
                                Text(
                                    text = "Connected",
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.White
                                )
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = status.gpu ?: status.device ?: "Ready",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        } else {
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = Color(0xFFEF4444)
                            ) {
                                Text(
                                    text = "Not Connected",
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.White
                                )
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = status.error ?: "Check server address",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.weight(1f))
                    
                    TextButton(onClick = {
                        onUrlChange(urlText)
                        onCheckServer()
                    }) {
                        Text("Test")
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onUrlChange(urlText)
                onDismiss()
            }) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

/**
 * Dialog showing upload/vectorization progress.
 */
@Composable
fun UploadProgressDialog(
    progress: UploadProgress,
    onCancel: () -> Unit
) {
    AlertDialog(
        onDismissRequest = { /* Don't allow dismiss while in progress */ },
        title = { Text("Processing Document") },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator(
                    progress = { progress.progress },
                    modifier = Modifier.size(64.dp),
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = progress.status,
                    style = MaterialTheme.typography.bodyMedium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                LinearProgressIndicator(
                    progress = { progress.progress },
                    modifier = Modifier.fillMaxWidth(),
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Text(
                    text = "${(progress.progress * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onCancel) {
                Text("Cancel")
            }
        }
    )
}
