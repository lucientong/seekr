//! Batch embedding wrapper.
//!
//! Groups input texts into batches and calls the underlying Embedder
//! for batch-optimized inference, improving throughput for index building.

use crate::embedder::traits::Embedder;
use crate::error::EmbedderError;

/// Batch embedding processor with progress reporting.
pub struct BatchEmbedder<E: Embedder> {
    embedder: E,
    batch_size: usize,
}

impl<E: Embedder> BatchEmbedder<E> {
    /// Create a new BatchEmbedder wrapping the given embedder.
    pub fn new(embedder: E, batch_size: usize) -> Self {
        Self {
            embedder,
            batch_size: batch_size.max(1),
        }
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    /// Embed all texts in batches, calling the progress callback after each batch.
    ///
    /// `progress_fn` receives (completed_count, total_count).
    pub fn embed_all_with_progress<F>(
        &self,
        texts: &[String],
        mut progress_fn: F,
    ) -> Result<Vec<Vec<f32>>, EmbedderError>
    where
        F: FnMut(usize, usize),
    {
        let total = texts.len();
        let mut all_embeddings = Vec::with_capacity(total);
        let mut completed = 0;

        for chunk in texts.chunks(self.batch_size) {
            let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let batch_result = self.embedder.embed_batch(&refs)?;
            all_embeddings.extend(batch_result);
            completed += chunk.len();
            progress_fn(completed, total);
        }

        Ok(all_embeddings)
    }

    /// Embed all texts in batches without progress reporting.
    pub fn embed_all(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_all_with_progress(texts, |_, _| {})
    }

    /// Get a reference to the inner embedder.
    pub fn inner(&self) -> &E {
        &self.embedder
    }
}

/// A dummy embedder for testing that produces random-like but deterministic vectors.
pub struct DummyEmbedder {
    dim: usize,
}

impl DummyEmbedder {
    /// Create a new dummy embedder with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for DummyEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        // Generate a deterministic pseudo-random embedding based on text content
        let mut embedding = vec![0.0f32; self.dim];
        let mut hash: u64 = 5381;

        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        for (i, val) in embedding.iter_mut().enumerate() {
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *val = ((hash >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            // Mix in position
            let _ = i; // suppress unused warning, position affects hash via iteration
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_embedder() {
        let embedder = DummyEmbedder::new(384);
        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check L2 norm ≈ 1.0
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2 normalized");
    }

    #[test]
    fn test_dummy_embedder_deterministic() {
        let embedder = DummyEmbedder::new(384);
        let e1 = embedder.embed("test").unwrap();
        let e2 = embedder.embed("test").unwrap();
        assert_eq!(e1, e2, "Same input should produce same embedding");
    }

    #[test]
    fn test_dummy_embedder_different_inputs() {
        let embedder = DummyEmbedder::new(384);
        let e1 = embedder.embed("hello").unwrap();
        let e2 = embedder.embed("world").unwrap();
        assert_ne!(e1, e2, "Different inputs should produce different embeddings");
    }

    #[test]
    fn test_batch_embedder() {
        let embedder = DummyEmbedder::new(128);
        let batch = BatchEmbedder::new(embedder, 2);

        let texts: Vec<String> = vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
            "bar".to_string(),
            "baz".to_string(),
        ];

        let mut progress_calls = Vec::new();
        let results = batch
            .embed_all_with_progress(&texts, |completed, total| {
                progress_calls.push((completed, total));
            })
            .unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].len(), 128);

        // Should have 3 progress calls (batches of 2, 2, 1)
        assert_eq!(progress_calls.len(), 3);
        assert_eq!(progress_calls[0], (2, 5));
        assert_eq!(progress_calls[1], (4, 5));
        assert_eq!(progress_calls[2], (5, 5));
    }
}
