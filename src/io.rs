//! Serialization and deserialization in genann wire format.

use std::io::{BufRead, Write};

use crate::activation::Activation;
use crate::ann::{compute_sizes, Ann};
use crate::error::{AnnError, Result};

impl Ann {
    /// Write the network to `w` in genann wire format.
    ///
    /// The output is a single newline-terminated line:
    ///
    /// ```text
    /// {inputs} {hidden_layers} {hidden} {outputs} {w0:.20e} {w1:.20e} … {wN:.20e}
    /// ```
    ///
    /// Activation functions are **not** written (not part of the genann
    /// format). Restore them manually after calling [`Ann::read`] if needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use runann::Ann;
    ///
    /// let ann = Ann::new(2, 1, 3, 1).unwrap();
    /// let mut buf = Vec::new();
    /// ann.write(&mut buf).unwrap();
    /// // buf now contains the serialized network as UTF-8 text
    /// ```
    pub fn write<W: Write>(&self, mut w: W) -> Result<()> {
        write!(
            w,
            "{} {} {} {}",
            self.inputs, self.hidden_layers, self.hidden, self.outputs
        )?;
        for &weight in &self.weights {
            write!(w, " {:.20e}", weight)?;
        }
        writeln!(w)?;
        Ok(())
    }

    /// Read a network from `r` in genann wire format.
    ///
    /// Parses the first line produced by [`Ann::write`].  The buffer is
    /// constructed **without** calling [`Ann::new`], so weights are taken
    /// verbatim from the file and `randomize` is never called.
    ///
    /// Both activation functions default to [`Activation::SigmoidCached`] after
    /// loading; set them explicitly if the original network used different ones.
    ///
    /// # Errors
    ///
    /// Returns [`AnnError::Io`] on read errors, [`AnnError::Parse`] if the
    /// line cannot be parsed, or [`AnnError::InvalidTopology`] if the stored
    /// topology is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::{BufReader, Cursor};
    /// use runann::Ann;
    ///
    /// let mut ann = Ann::new(2, 1, 3, 1).unwrap();
    /// let mut buf = Vec::new();
    /// ann.write(&mut buf).unwrap();
    ///
    /// let loaded = Ann::read(BufReader::new(Cursor::new(&buf))).unwrap();
    /// assert_eq!(ann.weights_ref(), loaded.weights_ref());
    /// ```
    pub fn read<R: BufRead>(mut r: R) -> Result<Ann> {
        let mut line = String::new();
        r.read_line(&mut line)?;
        let line = line.trim();

        let mut tokens = line.split_ascii_whitespace();

        macro_rules! next_usize {
            () => {
                tokens
                    .next()
                    .ok_or_else(|| AnnError::Parse("unexpected end of input".to_string()))
                    .and_then(|s| {
                        s.parse::<usize>()
                            .map_err(|e| AnnError::Parse(e.to_string()))
                    })
            };
        }

        let inputs = next_usize!()?;
        let hidden_layers = next_usize!()?;
        let hidden = next_usize!()?;
        let outputs = next_usize!()?;

        if inputs == 0 {
            return Err(AnnError::InvalidTopology("inputs must be >= 1".to_string()));
        }
        if outputs == 0 {
            return Err(AnnError::InvalidTopology(
                "outputs must be >= 1".to_string(),
            ));
        }
        if hidden_layers > 0 && hidden == 0 {
            return Err(AnnError::InvalidTopology(
                "hidden must be >= 1 when hidden_layers > 0".to_string(),
            ));
        }

        let (total_weights, total_neurons) = compute_sizes(inputs, hidden_layers, hidden, outputs);

        let mut weights = Vec::with_capacity(total_weights);
        for token in tokens {
            let w: f64 = token.parse()?;
            weights.push(w);
        }

        if weights.len() != total_weights {
            return Err(AnnError::Parse(format!(
                "expected {} weights, got {}",
                total_weights,
                weights.len()
            )));
        }

        Ok(Ann {
            inputs,
            hidden_layers,
            hidden,
            outputs,
            activation_hidden: Activation::default(),
            activation_output: Activation::default(),
            total_weights,
            total_neurons,
            weights,
            outputs_buf: vec![0.0; inputs + total_neurons],
            deltas: vec![0.0; total_neurons],
        })
    }
}
