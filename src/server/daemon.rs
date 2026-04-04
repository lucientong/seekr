//! Watch daemon for real-time incremental indexing.
//!
//! Monitors file system changes via the async watcher and triggers
//! incremental index updates with debounce to batch rapid changes.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::config::SeekrConfig;
use crate::embedder::traits::Embedder;
use crate::index::incremental::IncrementalState;
use crate::index::store::SeekrIndex;
use crate::scanner::watcher::{FileEvent, dedup_events, start_async_watcher};

/// Default debounce interval in milliseconds.
const DEFAULT_DEBOUNCE_MS: u64 = 500;

/// Run the watch daemon that monitors file changes and updates the index.
///
/// This function spawns an async task that:
/// 1. Listens for file system events via the async watcher
/// 2. Debounces rapid changes (batches events within a time window)
/// 3. Triggers incremental index updates for changed files
/// 4. Removes deleted files from the index
///
/// The `index` is shared with the HTTP server via `Arc<RwLock<>>`.
pub async fn run_watch_daemon(
    watch_path: &Path,
    config: &SeekrConfig,
    index: Arc<RwLock<SeekrIndex>>,
    debounce_ms: Option<u64>,
) -> Result<(), crate::error::ServerError> {
    let debounce = Duration::from_millis(debounce_ms.unwrap_or(DEFAULT_DEBOUNCE_MS));
    let watch_path = watch_path.to_path_buf();

    // Start the async file watcher
    let (_watcher, mut rx) = start_async_watcher(&watch_path)
        .map_err(|e| crate::error::ServerError::Internal(format!("Watch error: {}", e)))?;

    tracing::info!(
        path = %watch_path.display(),
        debounce_ms = debounce.as_millis() as u64,
        "Watch daemon started — monitoring for file changes"
    );

    // Load incremental state
    let index_dir = config.index_dir.join(
        watch_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .as_ref(),
    );
    let state_path = index_dir.join("incremental_state.json");
    let mut inc_state = IncrementalState::load(&state_path).unwrap_or_default();

    // Main event loop
    let mut pending_events: Vec<FileEvent> = Vec::new();

    loop {
        // Wait for the first event or timeout to process pending events
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(fe) => {
                        pending_events.push(fe);
                        // Drain any additional events that arrived
                        while let Ok(more) = rx.try_recv() {
                            pending_events.push(more);
                        }
                    }
                    None => {
                        tracing::warn!("File watcher channel closed, stopping daemon");
                        break;
                    }
                }

                // Start debounce timer — collect more events within the window
                tokio::time::sleep(debounce).await;

                // Drain any events that arrived during debounce
                while let Ok(more) = rx.try_recv() {
                    pending_events.push(more);
                }

                // Process the batch
                if !pending_events.is_empty() {
                    let events = std::mem::take(&mut pending_events);
                    let deduped = dedup_events(events);

                    match process_events(&deduped, &index, &mut inc_state, config).await {
                        Ok((added, removed)) => {
                            if added > 0 || removed > 0 {
                                tracing::info!(
                                    added = added,
                                    removed = removed,
                                    "Incremental index updated"
                                );

                                // Save incremental state
                                if let Err(e) = inc_state.save(&state_path) {
                                    tracing::warn!("Failed to save incremental state: {}", e);
                                }

                                // Save the index
                                let idx = index.read().await;
                                if let Err(e) = idx.save(&index_dir) {
                                    tracing::warn!("Failed to save index: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error processing file events: {}", e);
                            // Continue running despite errors
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Process a batch of deduplicated file events.
///
/// Returns (chunks_added, chunks_removed).
async fn process_events(
    events: &[FileEvent],
    index: &Arc<RwLock<SeekrIndex>>,
    inc_state: &mut IncrementalState,
    config: &SeekrConfig,
) -> Result<(usize, usize), String> {
    let mut changed_files: Vec<PathBuf> = Vec::new();
    let mut deleted_files: Vec<PathBuf> = Vec::new();

    for event in events {
        match event {
            FileEvent::Changed(path) => {
                // Filter by supported file extensions
                if is_supported_file(path) {
                    changed_files.push(path.clone());
                }
            }
            FileEvent::Deleted(path) => {
                deleted_files.push(path.clone());
            }
        }
    }

    let mut total_added = 0;
    let mut total_removed = 0;

    // Handle deletions
    if !deleted_files.is_empty() {
        let chunk_ids_to_remove = inc_state.chunk_ids_to_remove(&deleted_files);
        if !chunk_ids_to_remove.is_empty() {
            let mut idx = index.write().await;
            idx.remove_chunks(&chunk_ids_to_remove);
            total_removed = chunk_ids_to_remove.len();
        }
        inc_state.apply_deletions(&deleted_files);

        tracing::debug!(
            count = deleted_files.len(),
            chunks = total_removed,
            "Removed deleted files from index"
        );
    }

    // Handle changed/new files
    if !changed_files.is_empty() {
        // Remove old chunks for changed files first
        for file in &changed_files {
            let old_ids = inc_state.chunk_ids_for_file(file);
            if !old_ids.is_empty() {
                let mut idx = index.write().await;
                idx.remove_chunks(&old_ids);
                total_removed += old_ids.len();
            }
        }

        // Parse and re-index changed files
        let embedder = create_embedder(config)?;

        for file in &changed_files {
            match process_single_file(file, &*embedder, index, inc_state).await {
                Ok(count) => {
                    total_added += count;
                    tracing::debug!(file = %file.display(), chunks = count, "Re-indexed file");
                }
                Err(e) => {
                    tracing::warn!(file = %file.display(), error = %e, "Failed to index file");
                }
            }
        }
    }

    Ok((total_added, total_removed))
}

/// Parse, embed, and add a single file to the index.
///
/// Returns the number of chunks added.
async fn process_single_file(
    file: &Path,
    embedder: &dyn Embedder,
    index: &Arc<RwLock<SeekrIndex>>,
    inc_state: &mut IncrementalState,
) -> Result<usize, String> {
    // Read file content for hashing
    let content = std::fs::read(file).map_err(|e| e.to_string())?;

    // Parse into chunks using the existing chunker
    let parse_result =
        crate::parser::chunker::chunk_file_from_path(file).map_err(|e| e.to_string())?;

    let chunks = match parse_result {
        Some(result) => result.chunks,
        None => {
            // Unsupported language or binary file — still track in state
            inc_state.update_file(file.to_path_buf(), &content, Vec::new());
            return Ok(0);
        }
    };

    if chunks.is_empty() {
        inc_state.update_file(file.to_path_buf(), &content, Vec::new());
        return Ok(0);
    }

    // Generate embeddings
    let texts: Vec<&str> = chunks.iter().map(|c| c.body.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts).map_err(|e| e.to_string())?;

    // Add to index
    let mut chunk_ids = Vec::new();
    {
        let mut idx = index.write().await;
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let text_tokens = crate::index::store::tokenize_for_index_pub(&chunk.body);
            let entry = crate::index::IndexEntry {
                chunk_id: chunk.id,
                embedding: embedding.clone(),
                text_tokens,
            };
            idx.add_entry(entry, chunk.clone());
            chunk_ids.push(chunk.id);
        }
    }

    // Update incremental state
    inc_state.update_file(file.to_path_buf(), &content, chunk_ids);

    Ok(chunks.len())
}

/// Check if a file has a supported extension for indexing.
fn is_supported_file(path: &Path) -> bool {
    let supported = [
        "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "sh",
        "bash", "json", "toml", "yaml", "yml", "html", "css",
    ];

    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| supported.contains(&ext))
        .unwrap_or(false)
}

/// Create an embedder for the daemon.
fn create_embedder(config: &SeekrConfig) -> Result<Box<dyn Embedder>, String> {
    match crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir) {
        Ok(embedder) => Ok(Box::new(embedder)),
        Err(e) => Err(format!("Failed to create embedder: {}", e)),
    }
}
