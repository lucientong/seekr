//! File system watcher.
//!
//! Uses the `notify` crate to watch for file changes and
//! emit events through a tokio channel for incremental index updates.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use notify::{
    Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};

use crate::error::ScannerError;

/// A file system event suitable for incremental indexing.
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// A file was created or modified.
    Changed(PathBuf),
    /// A file was deleted.
    Deleted(PathBuf),
}

/// A file watcher that monitors a directory tree for changes.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    receiver: mpsc::Receiver<FileEvent>,
}

impl FileWatcher {
    /// Create a new file watcher for the given directory.
    ///
    /// Starts watching recursively. File events can be received
    /// via the `recv` method.
    pub fn new(watch_path: &Path) -> Result<Self, ScannerError> {
        let (tx, rx) = mpsc::channel();

        let sender = tx.clone();
        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        let file_events = convert_event(event);
                        for fe in file_events {
                            let _ = sender.send(fe);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("File watcher error: {}", e);
                    }
                }
            },
            Config::default().with_poll_interval(Duration::from_secs(2)),
        )
        .map_err(|e| ScannerError::WatchError(format!("Failed to create watcher: {}", e)))?;

        watcher
            .watch(watch_path, RecursiveMode::Recursive)
            .map_err(|e| {
                ScannerError::WatchError(format!(
                    "Failed to watch '{}': {}",
                    watch_path.display(),
                    e,
                ))
            })?;

        tracing::info!(path = %watch_path.display(), "File watcher started");

        Ok(Self {
            _watcher: watcher,
            receiver: rx,
        })
    }

    /// Receive the next file event, blocking until one is available.
    pub fn recv(&self) -> Option<FileEvent> {
        self.receiver.recv().ok()
    }

    /// Try to receive a file event without blocking.
    pub fn try_recv(&self) -> Option<FileEvent> {
        self.receiver.try_recv().ok()
    }

    /// Drain all pending events (non-blocking).
    pub fn drain_events(&self) -> Vec<FileEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.receiver.try_recv() {
            events.push(event);
        }
        events
    }

    /// Receive with timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<FileEvent> {
        self.receiver.recv_timeout(timeout).ok()
    }
}

/// Convert a notify Event into our FileEvent(s).
fn convert_event(event: Event) -> Vec<FileEvent> {
    let mut file_events = Vec::new();

    match event.kind {
        EventKind::Create(_) | EventKind::Modify(_) => {
            for path in event.paths {
                if path.is_file() {
                    file_events.push(FileEvent::Changed(path));
                }
            }
        }
        EventKind::Remove(_) => {
            for path in event.paths {
                file_events.push(FileEvent::Deleted(path));
            }
        }
        _ => {
            // Ignore access, other events
        }
    }

    file_events
}

/// Create an async watcher that sends events through a tokio channel.
///
/// This is useful for integration with the async Daemon mode.
pub fn start_async_watcher(
    watch_path: &Path,
) -> Result<
    (
        RecommendedWatcher,
        tokio::sync::mpsc::UnboundedReceiver<FileEvent>,
    ),
    ScannerError,
> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let sender = tx;
    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    let file_events = convert_event(event);
                    for fe in file_events {
                        let _ = sender.send(fe);
                    }
                }
                Err(e) => {
                    tracing::warn!("File watcher error: {}", e);
                }
            }
        },
        Config::default().with_poll_interval(Duration::from_secs(2)),
    )
    .map_err(|e| ScannerError::WatchError(format!("Failed to create async watcher: {}", e)))?;

    watcher
        .watch(watch_path, RecursiveMode::Recursive)
        .map_err(|e| {
            ScannerError::WatchError(format!(
                "Failed to watch '{}': {}",
                watch_path.display(),
                e,
            ))
        })?;

    tracing::info!(path = %watch_path.display(), "Async file watcher started");

    Ok((watcher, rx))
}

/// Deduplicate events by path, keeping only the latest event type per path.
pub fn dedup_events(events: Vec<FileEvent>) -> Vec<FileEvent> {
    use std::collections::HashMap;

    let mut latest: HashMap<PathBuf, FileEvent> = HashMap::new();

    for event in events {
        let path = match &event {
            FileEvent::Changed(p) => p.clone(),
            FileEvent::Deleted(p) => p.clone(),
        };
        latest.insert(path, event);
    }

    latest.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_events() {
        let events = vec![
            FileEvent::Changed(PathBuf::from("/a/b.rs")),
            FileEvent::Changed(PathBuf::from("/a/b.rs")),
            FileEvent::Deleted(PathBuf::from("/a/c.rs")),
            FileEvent::Changed(PathBuf::from("/a/c.rs")),
        ];

        let deduped = dedup_events(events);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_convert_create_event() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![file_path.clone()],
            attrs: Default::default(),
        };

        let file_events = convert_event(event);
        assert_eq!(file_events.len(), 1);
        match &file_events[0] {
            FileEvent::Changed(p) => assert_eq!(p, &file_path),
            _ => panic!("Expected Changed event"),
        }
    }

    #[test]
    fn test_convert_remove_event() {
        let event = Event {
            kind: EventKind::Remove(notify::event::RemoveKind::File),
            paths: vec![PathBuf::from("/a/deleted.rs")],
            attrs: Default::default(),
        };

        let file_events = convert_event(event);
        assert_eq!(file_events.len(), 1);
        match &file_events[0] {
            FileEvent::Deleted(p) => assert_eq!(p, &PathBuf::from("/a/deleted.rs")),
            _ => panic!("Expected Deleted event"),
        }
    }

    #[test]
    fn test_watcher_creation() {
        let dir = tempfile::tempdir().unwrap();
        let watcher = FileWatcher::new(dir.path());
        assert!(watcher.is_ok(), "Should be able to create a watcher");
    }
}
