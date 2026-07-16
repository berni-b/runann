//! Error types for runann.

use std::fmt;
use std::num::ParseFloatError;

/// Errors that can occur when creating, training, or serializing a network.
#[derive(Debug)]
pub enum AnnError {
    /// The requested network topology is invalid (e.g. zero inputs, or hidden
    /// layers requested but `hidden == 0`).
    InvalidTopology(String),
    /// An I/O error occurred while reading or writing a network.
    Io(std::io::Error),
    /// A value could not be parsed while reading a saved network.
    Parse(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AnnError>;

impl fmt::Display for AnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnnError::InvalidTopology(msg) => write!(f, "invalid topology: {msg}"),
            AnnError::Io(e) => write!(f, "I/O error: {e}"),
            AnnError::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for AnnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AnnError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AnnError {
    fn from(e: std::io::Error) -> Self {
        AnnError::Io(e)
    }
}

impl From<ParseFloatError> for AnnError {
    fn from(e: ParseFloatError) -> Self {
        AnnError::Parse(e.to_string())
    }
}
