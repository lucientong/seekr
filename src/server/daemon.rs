//! Watch daemon for real-time incremental indexing.
//!
//! Monitors file system changes via the async watcher and triggers
//! incremental index updates with debounce to batch rapid changes.

use std::sync::Arc;
use std::time::Duration;

use crate::scanner::watcher::{dedup_events, start_async_watcher};
use crate::server::state::ProjectEngine;

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
    engine: Arc<ProjectEngine>,
    debounce_ms: Option<u64>,
) -> Result<(), crate::error::ServerError> {
    let debounce = Duration::from_millis(debounce_ms.unwrap_or(DEFAULT_DEBOUNCE_MS));
    let watch_path = engine.project_path().to_path_buf();

    // Start the async file watcher
    let (_watcher, mut rx) = start_async_watcher(&watch_path)
        .map_err(|e| crate::error::ServerError::Internal(format!("Watch error: {}", e)))?;

    tracing::info!(
        path = %watch_path.display(),
        debounce_ms = debounce.as_millis() as u64,
        "Watch daemon started — monitoring for file changes"
    );

    let mut pending_events = Vec::new();

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

                    match Arc::clone(&engine).build_async(false).await {
                        Ok(report) => {
                            if report.changed_files > 0 || report.deleted_files > 0 {
                                tracing::info!(
                                    changed_files = report.changed_files,
                                    deleted_files = report.deleted_files,
                                    chunks = report.chunk_count,
                                    events = deduped.len(),
                                    "Watch index updated"
                                );
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
