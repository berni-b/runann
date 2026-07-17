//! # runann
//!
//! A minimal, dependency-light feed-forward neural network library — a pure-Rust
//! port of [genann](https://github.com/codeplea/genann).
//!
//! ## Features
//!
//! - Fully-connected feed-forward networks of arbitrary depth
//! - Online backpropagation training
//! - Five built-in activation functions (sigmoid, cached sigmoid, linear, threshold, ReLU)
//! - Save / load in genann's text wire format
//! - No `unsafe` code; only dependency is [`rand`](https://docs.rs/rand)
//!
//! ## Quick start
//!
//! ```rust
//! use runann::{Activation, Ann};
//!
//! // 2 inputs → 1 hidden layer of 3 neurons → 1 output
//! let mut ann = Ann::new(2, 1, 3, 1).unwrap();
//!
//! // Training data: XOR gate
//! let inputs  = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let targets = [[0.0],      [1.0],       [1.0],       [0.0]];
//!
//! for _ in 0..5_000 {
//!     for (inp, tgt) in inputs.iter().zip(targets.iter()) {
//!         ann.train(inp, tgt, 0.5);
//!     }
//! }
//!
//! for (inp, tgt) in inputs.iter().zip(targets.iter()) {
//!     let out = ann.run(inp);
//!     println!("XOR({}, {}) ≈ {:.3}  (expected {})", inp[0], inp[1], out[0], tgt[0]);
//! }
//! ```
//!
//! ## Weight layout
//!
//! Weights are stored in a flat `Vec<f64>`, layer by layer, neuron by neuron.
//! For each neuron the **bias weight comes first**, followed by one weight per
//! incoming connection:
//!
//! ```text
//! [ bias₀ | w₀₀ w₀₁ … | bias₁ | w₁₀ w₁₁ … | … ]
//! ```
//!
//! During the forward pass the bias contributes as `−bias_weight` (i.e. the
//! stored value is multiplied by `−1`), matching the genann convention.
//!
//! ## Saving and loading
//!
//! Networks can be round-tripped through any [`Write`](std::io::Write) /
//! [`BufRead`](std::io::BufRead) pair using [`Ann::write`] and [`Ann::read`].
//! The format is identical to genann's: a single space-separated line
//! `inputs hidden_layers hidden outputs w0 w1 … wN`.
//!
//! Activation functions are **not** persisted (not part of the genann format).
//! After loading, both activations default to [`Activation::SigmoidCached`];
//! restore non-default activations manually if needed.
//!
//! ## CLI binary
//!
//! This crate also ships a `runann` binary that loads a YAML config file and
//! provides `train` and `run` subcommands.  See the [`cli`] module and the
//! project README for details.

pub mod activation;
pub mod cli;
pub mod ann;
pub mod error;
mod forward;
mod io;
mod train;

pub use activation::Activation;
pub use ann::Ann;
pub use error::{AnnError, Result};
