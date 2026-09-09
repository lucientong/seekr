use std::io::Write;
use std::path::Path;

use crate::error::IndexError;

pub(crate) fn atomic_write(path: &Path, data: &[u8]) -> Result<(), IndexError> {
    let parent = path
        .parent()
        .ok_or_else(|| IndexError::Corrupted(format!("Path has no parent: {}", path.display())))?;
    std::fs::create_dir_all(parent)?;

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("index");
    let temp_path = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));

    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);

        #[cfg(windows)]
        if path.exists() {
            // std::fs::rename cannot replace an existing file on Windows.
            std::fs::remove_file(path)?;
        }
        std::fs::rename(&temp_path, path)?;

        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;

        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}
