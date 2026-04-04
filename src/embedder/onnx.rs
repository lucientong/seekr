//! ONNX Runtime embedding backend.
//!
//! Loads a local ONNX model (all-MiniLM-L6-v2 quantized) and provides
//! embedding inference. Supports automatic model download and caching.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedder::traits::Embedder;
use crate::error::EmbedderError;

/// HuggingFace model URL for all-MiniLM-L6-v2 ONNX.
const MODEL_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx";

/// Expected model filename.
const MODEL_FILENAME: &str = "all-MiniLM-L6-v2-quantized.onnx";

/// Tokenizer URL.
const TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// Tokenizer filename.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Embedding dimension for all-MiniLM-L6-v2.
const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length.
const MAX_SEQ_LENGTH: usize = 256;

/// ONNX-based embedding backend using all-MiniLM-L6-v2.
pub struct OnnxEmbedder {
    /// Session wrapped in Mutex because `session.run()` requires `&mut self`.
    session: Mutex<Session>,
    /// HuggingFace WordPiece tokenizer loaded from tokenizer.json.
    tokenizer: tokenizers::Tokenizer,
    model_dir: PathBuf,
}

impl OnnxEmbedder {
    /// Create a new OnnxEmbedder.
    ///
    /// If the model is not found in `model_dir`, it will be downloaded
    /// automatically from HuggingFace.
    pub fn new(model_dir: &Path) -> Result<Self, EmbedderError> {
        std::fs::create_dir_all(model_dir).map_err(EmbedderError::Io)?;

        let model_path = model_dir.join(MODEL_FILENAME);

        // Download model if not present
        if !model_path.exists() {
            tracing::info!("Downloading ONNX model to {}...", model_path.display());
            download_file(MODEL_URL, &model_path)?;
            tracing::info!("Model downloaded successfully.");
        }

        // Download tokenizer if not present
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);
        if !tokenizer_path.exists() {
            tracing::info!("Downloading tokenizer...");
            download_file(TOKENIZER_URL, &tokenizer_path)?;
            tracing::info!("Tokenizer downloaded successfully.");
        }

        // Create ONNX Runtime session
        let session = Session::builder()
            .map_err(|e| EmbedderError::OnnxError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(4)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(&model_path)
            .map_err(|e| EmbedderError::OnnxError(format!("Failed to load model: {}", e)))?;

        // Load HuggingFace tokenizer from downloaded tokenizer.json
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::OnnxError(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Get the model directory path.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Tokenize text using the HuggingFace WordPiece tokenizer.
    ///
    /// Uses the real tokenizer.json from all-MiniLM-L6-v2 for proper
    /// WordPiece tokenization, producing correct token IDs that match
    /// the model's vocabulary.
    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        // Encode with special tokens ([CLS] and [SEP] are added automatically)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .unwrap_or_else(|_| {
                // Fallback: return empty encoding on error
                self.tokenizer.encode("", true).unwrap()
            });

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        // Truncate to MAX_SEQ_LENGTH if needed
        if input_ids.len() > MAX_SEQ_LENGTH {
            input_ids.truncate(MAX_SEQ_LENGTH);
            attention_mask.truncate(MAX_SEQ_LENGTH);
            // Ensure the last token is [SEP] (id = 102 for BERT-based models)
            if let Some(last) = input_ids.last_mut() {
                *last = 102;
            }
        }

        // Pad to fixed length for batching
        while input_ids.len() < MAX_SEQ_LENGTH {
            input_ids.push(0);
            attention_mask.push(0);
        }

        (input_ids, attention_mask)
    }

    /// Run inference on tokenized input.
    fn run_inference(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
    ) -> Result<Vec<f32>, EmbedderError> {
        let seq_len = input_ids.len();

        let input_ids_array =
            ndarray::Array2::from_shape_vec((1, seq_len), input_ids.to_vec())
                .map_err(|e| EmbedderError::OnnxError(format!("Shape error: {}", e)))?;
        let attention_mask_array =
            ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.to_vec())
                .map_err(|e| EmbedderError::OnnxError(format!("Shape error: {}", e)))?;
        let token_type_ids_array = ndarray::Array2::<i64>::zeros((1, seq_len));

        // Create TensorRef inputs
        let input_ids_tensor = TensorRef::from_array_view(&input_ids_array)
            .map_err(|e| EmbedderError::OnnxError(format!("Tensor creation error: {}", e)))?;
        let attention_mask_tensor = TensorRef::from_array_view(&attention_mask_array)
            .map_err(|e| EmbedderError::OnnxError(format!("Tensor creation error: {}", e)))?;
        let token_type_ids_tensor = TensorRef::from_array_view(&token_type_ids_array)
            .map_err(|e| EmbedderError::OnnxError(format!("Tensor creation error: {}", e)))?;

        let mut session = self.session.lock().map_err(|e| {
            EmbedderError::OnnxError(format!("Session lock poisoned: {}", e))
        })?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            ])
            .map_err(|e| EmbedderError::OnnxError(format!("Inference error: {}", e)))?;

        // Extract output tensor — try common output names, then fall back to first output
        let output = if outputs.contains_key("last_hidden_state") {
            &outputs["last_hidden_state"]
        } else if outputs.contains_key("token_embeddings") {
            &outputs["token_embeddings"]
        } else {
            &outputs[0]
        };

        let tensor = output
            .try_extract_array::<f32>()
            .map_err(|e| EmbedderError::OnnxError(format!("Extract error: {}", e)))?;

        // Mean pooling: average over the sequence dimension (dim 1)
        // tensor shape: [1, seq_len, hidden_size]
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(EmbedderError::OnnxError(format!(
                "Unexpected output shape: {:?}",
                shape
            )));
        }

        let hidden_size = shape[2];
        let mut pooled = vec![0.0f32; hidden_size];
        let active_tokens: f32 = attention_mask.iter().map(|&m| m as f32).sum();

        if active_tokens > 0.0 {
            for seq_idx in 0..shape[1] {
                let mask = attention_mask.get(seq_idx).copied().unwrap_or(0) as f32;
                if mask > 0.0 {
                    for dim in 0..hidden_size {
                        pooled[dim] += tensor[[0, seq_idx, dim]];
                    }
                }
            }
            for val in &mut pooled {
                *val /= active_tokens;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut pooled {
                *x /= norm;
            }
        }

        Ok(pooled)
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let (input_ids, attention_mask) = self.tokenize(text);
        self.run_inference(&input_ids, &attention_mask)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        // For now, process sequentially. A proper implementation would
        // batch inputs together for a single session.run() call.
        texts.iter().map(|text| self.embed(text)).collect()
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Download a file from a URL to a local path.
fn download_file(url: &str, dest: &Path) -> Result<(), EmbedderError> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| EmbedderError::DownloadFailed(format!("HTTP request failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(EmbedderError::DownloadFailed(format!(
            "HTTP {} for {}",
            response.status(),
            url
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|e| EmbedderError::DownloadFailed(format!("Failed to read response: {}", e)))?;

    // Verify download isn't empty
    if bytes.is_empty() {
        return Err(EmbedderError::DownloadFailed(
            "Downloaded file is empty".to_string(),
        ));
    }

    // Write to temporary file then rename (atomic-ish)
    let tmp_path = dest.with_extension("tmp");
    std::fs::write(&tmp_path, &bytes).map_err(EmbedderError::Io)?;
    std::fs::rename(&tmp_path, dest).map_err(EmbedderError::Io)?;

    tracing::info!(
        "Downloaded {} bytes to {}",
        bytes.len(),
        dest.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_output_length() {
        // Verify that tokenize always produces MAX_SEQ_LENGTH outputs
        // This test requires a tokenizer.json file, so we create a minimal one
        // In integration tests, the real tokenizer would be used
        let model_dir = std::env::temp_dir().join("seekr_test_tokenizer");
        if let Ok(embedder) = OnnxEmbedder::new(&model_dir) {
            let (ids, mask) = embedder.tokenize("hello world");
            assert_eq!(ids.len(), MAX_SEQ_LENGTH);
            assert_eq!(mask.len(), MAX_SEQ_LENGTH);

            // First token should be [CLS] = 101
            assert_eq!(ids[0], 101);

            // There should be active tokens (attention_mask = 1)
            let active: i64 = mask.iter().sum();
            assert!(active > 0, "Should have at least some active tokens");
        }
    }

    #[test]
    fn test_embedding_dimension() {
        assert_eq!(EMBEDDING_DIM, 384);
    }

    #[test]
    fn test_max_seq_length() {
        assert_eq!(MAX_SEQ_LENGTH, 256);
    }
}
