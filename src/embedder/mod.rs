//! Embedding engine module.
//!
//! Provides trait-based embedding abstraction with ONNX Runtime backend.
//! Supports batch embedding computation for throughput optimization.

pub mod batch;
pub mod onnx;
pub mod traits;
