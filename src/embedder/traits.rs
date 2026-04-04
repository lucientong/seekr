//! Embedder trait definition.
//!
//! Defines the pluggable backend interface for text embedding.

use crate::error::EmbedderError;

/// Trait for text embedding backends.
///
/// Implementations convert text strings into dense vector representations
/// suitable for semantic similarity search.
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;

    /// Embed a batch of text strings into vectors.
    ///
    /// Default implementation calls `embed` for each text sequentially.
    /// Backends should override this for batch-optimized inference.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return the dimensionality of the output embedding vectors.
    fn dimension(&self) -> usize;
}

/// Blanket implementation for boxed trait objects.
impl Embedder for Box<dyn Embedder> {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        (**self).embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        (**self).embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }
}
