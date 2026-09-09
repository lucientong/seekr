//! Bounded, expiring session state for suppressing repeated search chunks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_TTL: Duration = Duration::from_secs(30 * 60);
const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_CHUNKS_PER_SESSION: usize = 1_000;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SessionKey {
    project: PathBuf,
    session_id: String,
}

struct SessionEntry {
    seen: HashSet<u64>,
    order: VecDeque<u64>,
    last_access: Instant,
}

struct StoreState {
    sessions: HashMap<SessionKey, SessionEntry>,
}

/// Thread-safe session cache shared by HTTP and MCP entry points.
#[derive(Clone)]
pub struct SessionDedupStore {
    state: Arc<Mutex<StoreState>>,
    ttl: Duration,
    max_sessions: usize,
    max_chunks_per_session: usize,
}

impl Default for SessionDedupStore {
    fn default() -> Self {
        Self::with_limits(
            DEFAULT_TTL,
            DEFAULT_MAX_SESSIONS,
            DEFAULT_MAX_CHUNKS_PER_SESSION,
        )
    }
}

impl SessionDedupStore {
    fn with_limits(ttl: Duration, max_sessions: usize, max_chunks_per_session: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(StoreState {
                sessions: HashMap::new(),
            })),
            ttl,
            max_sessions: max_sessions.max(1),
            max_chunks_per_session: max_chunks_per_session.max(1),
        }
    }

    /// Return a snapshot of chunk IDs already emitted for this project/session.
    pub fn seen_chunk_ids(&self, project: &Path, session_id: &str) -> HashSet<u64> {
        let now = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            return HashSet::new();
        };
        self.remove_expired(&mut state, now);
        let key = session_key(project, session_id);
        state
            .sessions
            .get_mut(&key)
            .map(|entry| {
                entry.last_access = now;
                entry.seen.clone()
            })
            .unwrap_or_default()
    }

    /// Record only chunks actually returned after result and token limits.
    pub fn record_returned(
        &self,
        project: &Path,
        session_id: &str,
        chunk_ids: impl IntoIterator<Item = u64>,
    ) {
        let chunk_ids: Vec<u64> = chunk_ids.into_iter().collect();
        if chunk_ids.is_empty() {
            return;
        }
        let now = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        self.remove_expired(&mut state, now);
        let key = session_key(project, session_id);
        if !state.sessions.contains_key(&key) && state.sessions.len() >= self.max_sessions {
            if let Some(oldest) = state
                .sessions
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| key.clone())
            {
                state.sessions.remove(&oldest);
            }
        }
        let entry = state.sessions.entry(key).or_insert_with(|| SessionEntry {
            seen: HashSet::new(),
            order: VecDeque::new(),
            last_access: now,
        });
        entry.last_access = now;
        for chunk_id in chunk_ids {
            if entry.seen.insert(chunk_id) {
                entry.order.push_back(chunk_id);
            }
            while entry.order.len() > self.max_chunks_per_session {
                if let Some(expired_id) = entry.order.pop_front() {
                    entry.seen.remove(&expired_id);
                }
            }
        }
    }

    fn remove_expired(&self, state: &mut StoreState, now: Instant) {
        state
            .sessions
            .retain(|_, entry| now.duration_since(entry.last_access) <= self.ttl);
    }
}

fn session_key(project: &Path, session_id: &str) -> SessionKey {
    SessionKey {
        project: project
            .canonicalize()
            .unwrap_or_else(|_| project.to_path_buf()),
        session_id: session_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolates_projects_and_evicts_old_chunks() {
        let store = SessionDedupStore::with_limits(Duration::from_secs(60), 4, 2);
        let project_a = Path::new("/tmp/seekr-a");
        let project_b = Path::new("/tmp/seekr-b");
        store.record_returned(project_a, "session", [1, 2, 3]);

        assert_eq!(
            store.seen_chunk_ids(project_a, "session"),
            HashSet::from([2, 3])
        );
        assert!(store.seen_chunk_ids(project_b, "session").is_empty());
    }

    #[test]
    fn evicts_least_recently_used_session_at_capacity() {
        let store = SessionDedupStore::with_limits(Duration::from_secs(60), 2, 10);
        let project = Path::new("/tmp/seekr");
        store.record_returned(project, "first", [1]);
        store.record_returned(project, "second", [2]);
        let _ = store.seen_chunk_ids(project, "second");
        store.record_returned(project, "third", [3]);

        assert!(store.seen_chunk_ids(project, "first").is_empty());
        assert_eq!(store.seen_chunk_ids(project, "second"), HashSet::from([2]));
    }

    #[test]
    fn expires_inactive_sessions() {
        let store = SessionDedupStore::with_limits(Duration::ZERO, 2, 10);
        let project = Path::new("/tmp/seekr");
        store.record_returned(project, "session", [1]);
        std::thread::sleep(Duration::from_millis(1));

        assert!(store.seen_chunk_ids(project, "session").is_empty());
    }
}
