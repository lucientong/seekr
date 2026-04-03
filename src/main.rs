//! seekr-code — A semantic code search engine, smarter than grep.
//!
//! Usage:
//!   seekr-code search "query"     Search code in the current project
//!   seekr-code index [path]       Build search index for a project
//!   seekr-code serve              Start HTTP API + MCP server
//!   seekr-code status             Show index status

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

/// Seekr: A semantic code search engine, smarter than grep.
///
/// Supports text regex + semantic vector + AST pattern search modes.
/// 100% local — no data leaves your machine.
#[derive(Parser)]
#[command(name = "seekr-code")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output results as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Verbose output (can be repeated: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Search code in the current project
    Search {
        /// Search query (text, semantic, or AST pattern)
        query: String,

        /// Search mode: text, semantic, ast, or hybrid (default)
        #[arg(short, long, default_value = "hybrid")]
        mode: String,

        /// Maximum number of results
        #[arg(short = 'k', long, default_value = "20")]
        top_k: usize,

        /// Project path to search in
        #[arg(short, long, default_value = ".")]
        path: String,
    },

    /// Build search index for a project
    Index {
        /// Project path to index (default: current directory)
        #[arg(default_value = ".")]
        path: String,

        /// Force full re-index, ignoring incremental state
        #[arg(long)]
        force: bool,
    },

    /// Start HTTP API + MCP server (daemon mode)
    Serve {
        /// Host address to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port number
        #[arg(short, long, default_value = "7720")]
        port: u16,

        /// Run as MCP server over stdio instead of HTTP
        #[arg(long)]
        mcp: bool,
    },

    /// Show index status for a project
    Status {
        /// Project path to check (default: current directory)
        #[arg(default_value = ".")]
        path: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing subscriber.
    // Priority: SEEKR_LOG env var > RUST_LOG env var > default based on verbosity
    let default_filter = match cli.verbose {
        0 => "seekr_code=warn",
        1 => "seekr_code=info",
        2 => "seekr_code=debug",
        _ => "seekr_code=trace",
    };

    let env_filter = EnvFilter::try_from_env("SEEKR_LOG")
        .or_else(|_| EnvFilter::try_from_env("RUST_LOG"))
        .unwrap_or_else(|_| EnvFilter::new(default_filter));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    // Load configuration
    let config = seekr_code::config::SeekrConfig::load()
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    match cli.command {
        Commands::Search {
            query,
            mode,
            top_k,
            path,
        } => {
            tracing::info!(query = %query, mode = %mode, top_k = top_k, path = %path, "Starting search");
            seekr_code::server::cli::cmd_search(&query, &mode, top_k, &path, &config, cli.json)?;
        }
        Commands::Index { path, force } => {
            tracing::info!(path = %path, force = force, "Building index");
            seekr_code::server::cli::cmd_index(&path, force, &config, cli.json)?;
        }
        Commands::Serve { host, port, mcp } => {
            if mcp {
                tracing::info!("Starting MCP server on stdio");
                seekr_code::server::mcp::run_mcp_stdio(&config)?;
            } else {
                tracing::info!(host = %host, port = port, "Starting HTTP server");
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?;
                rt.block_on(async {
                    seekr_code::server::http::start_http_server(&host, port, config).await
                })?;
            }
        }
        Commands::Status { path } => {
            tracing::info!(path = %path, "Checking status");
            seekr_code::server::cli::cmd_status(&path, &config, cli.json)?;
        }
    }

    Ok(())
}
