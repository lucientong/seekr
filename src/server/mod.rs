//! Server module.
//!
//! Provides CLI command implementations, HTTP REST API, MCP Server protocol,
//! and watch daemon for real-time incremental indexing.

pub mod cli;
pub mod daemon;
pub mod http;
pub mod mcp;
