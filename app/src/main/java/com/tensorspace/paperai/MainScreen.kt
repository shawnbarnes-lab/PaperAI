package com.tensorspace.paperai

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
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
import androidx.compose.ui.draw.alpha
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
 *
 * CHANGES FROM ORIGINAL:
 * - ResponseTab now shows brief summary + source passages (hybrid layout)
 * - SourcesTab cards show context window (±2 chunks) on expand
 * - DocumentViewerDialog uses proximity-based opacity for context chunks
 * - Added ExpandableSourceCard with inline context viewer
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

// =============================================================================
// STATUS BAR
// =============================================================================

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

// =============================================================================
// SEARCH BAR
// =============================================================================

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

// =============================================================================
// TAB SECTION — AI Response + Sources
// =============================================================================

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

// =============================================================================
// RESPONSE TAB — Now shows hybrid: brief summary + source passages
// =============================================================================

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
                // Query echo
                Text(
                    text = "You asked: \"${response.query}\"",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Brief summary card (if LLM produced one)
                val briefSummary = response.briefSummary
                if (briefSummary != null) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                text = "AI Summary",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = briefSummary,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Key source passages
                val sources = response.sources
                if (sources.isNotEmpty()) {
                    Text(
                        text = "📖 Key Passages",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    sources.take(3).forEachIndexed { index, chunk ->
                        SourcePassageCard(
                            index = index + 1,
                            chunk = chunk
                        )
                        if (index < minOf(2, sources.lastIndex)) {
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }

                    if (sources.size > 3) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "${sources.size - 3} more matching sections in Sources tab →",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontStyle = FontStyle.Italic
                        )
                    }
                }

                // Fallback: show full response text if no brief summary
                if (briefSummary == null) {
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
                                text = response.answer,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
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
 * Compact source passage card shown in the AI Response tab.
 * Shows a truncated preview of the matched chunk with source info.
 */
@Composable
fun SourcePassageCard(
    index: Int,
    chunk: DocumentChunk
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            // Preview text (first 200 chars)
            val preview = chunk.text.take(200).let {
                if (chunk.text.length > 200) "$it..." else it
            }
            Text(
                text = "[$index] $preview",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(modifier = Modifier.height(4.dp))
            // Source attribution
            val source = if (chunk.sectionTitle.isNotBlank()) {
                "${chunk.sourceName} — ${chunk.sectionTitle}, p.${chunk.pageNumber}"
            } else {
                "${chunk.sourceName}, p.${chunk.pageNumber}"
            }
            Text(
                text = source,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
                fontStyle = FontStyle.Italic
            )
        }
    }
}

fun formatProcessingTime(timeMs: Long): String {
    return when {
        timeMs < 1000 -> "${timeMs}ms"
        else -> String.format("%.1fs", timeMs / 1000.0)
    }
}

// =============================================================================
// SOURCES TAB — Now with expandable context windows
// =============================================================================

@Composable
fun SourcesTab(
    uiState: UiState,
    onSourceClick: (DocumentChunk) -> Unit = {}
) {
    val sources = uiState.lastResponse?.sources ?: emptyList()
    val contextChunks = uiState.lastResponse?.contextChunks ?: emptyMap()

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
                ExpandableSourceCard(
                    chunk = chunk,
                    contextWindow = contextChunks[chunk.id] ?: emptyList(),
                    onViewDocument = { onSourceClick(chunk) }
                )
            }
        }
    }
}

/**
 * Expandable source card — shows the matched chunk, and expands to show
 * ±2 surrounding chunks as context when tapped.
 *
 * This solves the "chunks are cut off" problem without needing to open the
 * full document viewer. Users can see surrounding paragraphs inline.
 */
@Composable
fun ExpandableSourceCard(
    chunk: DocumentChunk,
    contextWindow: List<DocumentChunk>,
    onViewDocument: () -> Unit = {}
) {
    var isExpanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { isExpanded = !isExpanded },
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header: source name + relevance score
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
                    val pageInfo = if (chunk.sectionTitle.isNotBlank()) {
                        "Page ${chunk.pageNumber} — ${chunk.sectionTitle}"
                    } else {
                        "Page ${chunk.pageNumber}"
                    }
                    Text(
                        text = pageInfo,
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

            // Matched chunk text
            Text(
                text = chunk.text,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )

            // Expanded context window
            if (isExpanded && contextWindow.size > 1) {
                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Surrounding Context",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(8.dp))

                contextWindow.forEach { contextChunk ->
                    if (contextChunk.id == chunk.id) return@forEach  // skip the match itself

                    val isBefore = contextChunk.chunkIndex < chunk.chunkIndex ||
                            (contextChunk.chunkIndex == 0 && contextChunk.pageNumber < chunk.pageNumber)
                    val label = if (isBefore) "↑ Before" else "↓ After"

                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        shape = RoundedCornerShape(6.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = "$label (p.${contextChunk.pageNumber})",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                fontStyle = FontStyle.Italic
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = contextChunk.text,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Footer: expand hint + view full doc
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = if (isExpanded) "Tap to collapse" else {
                        val extraCount = contextWindow.size - 1
                        if (extraCount > 0) "Tap to show $extraCount surrounding sections"
                        else getRelevanceLabel(chunk.similarity)
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontStyle = FontStyle.Italic
                )
                TextButton(
                    onClick = onViewDocument,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                ) {
                    Text(
                        text = "View full document →",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

// =============================================================================
// DOCUMENT VIEWER DIALOG — Improved with proximity-based opacity
// =============================================================================

/**
 * Full-screen dialog to view an entire document.
 * Highlights the matched chunk and uses proximity-based opacity
 * so context chunks near the match are brighter, distant chunks are dimmer.
 */
@Composable
fun DocumentViewerDialog(
    documentName: String,
    chunks: List<DocumentChunk>,
    highlightedChunkId: Long? = null,
    onDismiss: () -> Unit
) {
    val sortedChunks = remember(chunks) {
        chunks.sortedBy { it.chunkIndex.takeIf { idx -> idx > 0 } ?: it.pageNumber }
    }
    val listState = rememberLazyListState()

    // Auto-scroll to highlighted chunk when dialog opens
    LaunchedEffect(highlightedChunkId, sortedChunks) {
        if (highlightedChunkId != null) {
            val index = sortedChunks.indexOfFirst { it.id == highlightedChunkId }
            if (index >= 0) {
                // Scroll so the match is near the top with some leading context
                val scrollTarget = (index - 1).coerceAtLeast(0)
                listState.animateScrollToItem(scrollTarget)
            }
        }
    }

    // Compute the index of the highlighted chunk for proximity calculations
    val highlightIndex = remember(highlightedChunkId, sortedChunks) {
        sortedChunks.indexOfFirst { it.id == highlightedChunkId }
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
                        itemsIndexed(sortedChunks) { index, chunk ->
                            val isHighlighted = chunk.id == highlightedChunkId
                            val distanceFromMatch = if (highlightIndex >= 0) {
                                kotlin.math.abs(index - highlightIndex)
                            } else {
                                Int.MAX_VALUE
                            }
                            // Proximity-based opacity: match=1.0, ±1=0.85, ±2=0.7, rest=0.5
                            val sectionAlpha = when (distanceFromMatch) {
                                0 -> 1f
                                1 -> 0.85f
                                2 -> 0.7f
                                else -> 0.5f
                            }

                            DocumentSection(
                                chunk = chunk,
                                isHighlighted = isHighlighted,
                                isNearContext = distanceFromMatch in 1..2,
                                alpha = sectionAlpha
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Individual document section within the document viewer.
 * Now supports proximity-based visual treatment.
 */
@Composable
fun DocumentSection(
    chunk: DocumentChunk,
    isHighlighted: Boolean = false,
    isNearContext: Boolean = false,
    alpha: Float = 1f
) {
    val backgroundColor by animateColorAsState(
        targetValue = when {
            isHighlighted -> MaterialTheme.colorScheme.tertiaryContainer
            isNearContext -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
            else -> Color.Transparent
        },
        animationSpec = tween(300),
        label = "sectionBgColor"
    )

    val borderColor = if (isHighlighted) {
        MaterialTheme.colorScheme.tertiary
    } else {
        Color.Transparent
    }

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .alpha(alpha),
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
                Column {
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
                    if (chunk.sectionTitle.isNotBlank()) {
                        Text(
                            text = chunk.sectionTitle,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
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
                } else if (isNearContext) {
                    Text(
                        text = "CONTEXT",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontStyle = FontStyle.Italic
                    )
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
            if (!isHighlighted && !isNearContext) {
                Spacer(modifier = Modifier.height(8.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            }
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

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

// =============================================================================
// INITIALIZING + WELCOME VIEWS (unchanged from original)
// =============================================================================

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

                        HowItWorksStep(number = "1", text = "Add your document (PDF)")
                        Spacer(modifier = Modifier.height(8.dp))
                        HowItWorksStep(number = "2", text = "We process it into a knowledge pack")
                        Spacer(modifier = Modifier.height(8.dp))
                        HowItWorksStep(number = "3", text = "Ask questions anytime - no internet needed")
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

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
                    OutlinedButton(
                        onClick = onLoadSamples,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text("Try Sample Content")
                    }

                    OutlinedButton(
                        onClick = onImport,
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text("Import Pack")
                    }
                }

                Spacer(modifier = Modifier.height(20.dp))

                Text(
                    text = "Already have a .pack file? Tap 'Import Pack' to load it instantly.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )

                Spacer(modifier = Modifier.height(24.dp))

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

// =============================================================================
// DIALOGS (unchanged from original)
// =============================================================================

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
