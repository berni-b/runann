//! CLI command implementations: `train` and `run`.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use crate::{Ann, AnnError};

use crate::cli::config::Config;

/// Errors that can occur in CLI commands.
#[derive(Debug)]
pub enum CliError {
    Ann(AnnError),
    Io(std::io::Error),
    Yaml(serde_yaml::Error),
    Config(String),
    Csv { line: usize, msg: String },
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Ann(e) => write!(f, "network error: {e}"),
            CliError::Io(e) => write!(f, "I/O error: {e}"),
            CliError::Yaml(e) => write!(f, "YAML error: {e}"),
            CliError::Config(msg) => write!(f, "config error: {msg}"),
            CliError::Csv { line, msg } => write!(f, "CSV error at line {line}: {msg}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Ann(e) => Some(e),
            CliError::Io(e) => Some(e),
            CliError::Yaml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<AnnError> for CliError {
    fn from(e: AnnError) -> Self {
        CliError::Ann(e)
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        CliError::Io(e)
    }
}

impl From<serde_yaml::Error> for CliError {
    fn from(e: serde_yaml::Error) -> Self {
        CliError::Yaml(e)
    }
}

/// Output format for inference results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Plain,
    Json,
}

/// Parse a CSV data file into input/output pairs.
///
/// - Lines starting with `#` or empty lines are skipped.
/// - Each row must have exactly `inputs + outputs` columns.
/// - Returns `(inputs_vec, outputs_vec)` pairs.
pub fn parse_csv(
    path: &str,
    inputs: usize,
    outputs: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, CliError> {
    let file = File::open(path).map_err(CliError::Io)?;
    parse_csv_reader(BufReader::new(file), inputs, outputs)
}

/// Parse CSV data from a reader.
pub fn parse_csv_reader<R: BufRead>(
    reader: R,
    inputs: usize,
    outputs: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, CliError> {
    let expected_cols = inputs + outputs;
    let mut rows = Vec::new();
    let mut line_num = 0usize;

    for line in reader.lines() {
        line_num += 1;
        let line = line.map_err(CliError::Io)?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != expected_cols {
            return Err(CliError::Csv {
                line: line_num,
                msg: format!(
                    "expected {} columns ({} inputs + {} outputs), got {}",
                    expected_cols,
                    inputs,
                    outputs,
                    parts.len()
                ),
            });
        }

        let mut input_vals = Vec::with_capacity(inputs);
        for (i, part) in parts[..inputs].iter().enumerate() {
            let v: f64 = part.trim().parse().map_err(|e| CliError::Csv {
                line: line_num,
                msg: format!("column {}: {e}", i + 1),
            })?;
            input_vals.push(v);
        }

        let mut output_vals = Vec::with_capacity(outputs);
        for (i, part) in parts[inputs..].iter().enumerate() {
            let v: f64 = part.trim().parse().map_err(|e| CliError::Csv {
                line: line_num,
                msg: format!("column {}: {e}", inputs + i + 1),
            })?;
            output_vals.push(v);
        }

        rows.push((input_vals, output_vals));
    }

    Ok(rows)
}

/// Parse only the input columns from a CSV file (for batch inference).
pub fn parse_csv_inputs<R: BufRead>(
    reader: R,
    inputs: usize,
) -> Result<Vec<Vec<f64>>, CliError> {
    let mut rows = Vec::new();
    let mut line_num = 0usize;

    for line in reader.lines() {
        line_num += 1;
        let line = line.map_err(CliError::Io)?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < inputs {
            return Err(CliError::Csv {
                line: line_num,
                msg: format!("expected at least {} columns, got {}", inputs, parts.len()),
            });
        }

        let mut input_vals = Vec::with_capacity(inputs);
        for (i, part) in parts[..inputs].iter().enumerate() {
            let v: f64 = part.trim().parse().map_err(|e| CliError::Csv {
                line: line_num,
                msg: format!("column {}: {e}", i + 1),
            })?;
            input_vals.push(v);
        }

        rows.push(input_vals);
    }

    Ok(rows)
}

/// Parse a single input sample from a comma-separated string.
pub fn parse_input_str(s: &str, inputs: usize) -> Result<Vec<f64>, CliError> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != inputs {
        return Err(CliError::Config(format!(
            "--input expects {inputs} values, got {}",
            parts.len()
        )));
    }
    parts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.trim().parse::<f64>().map_err(|e| CliError::Config(format!(
                "value {}: {e}",
                i + 1
            )))
        })
        .collect()
}

/// Format inference output values.
pub fn format_output(values: &[f64], format: OutputFormat) -> String {
    match format {
        OutputFormat::Plain => values
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(" "),
        OutputFormat::Json => {
            let inner = values
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
    }
}

/// Train a network from CSV data and save it to the model path in config.
pub fn run_train(config: &Config) -> Result<(), CliError> {
    let net_cfg = config.network.as_ref().ok_or_else(|| {
        CliError::Config("missing `network` block (required for train)".to_string())
    })?;
    let train_cfg = config.training.as_ref().ok_or_else(|| {
        CliError::Config("missing `training` block (required for train)".to_string())
    })?;
    let paths_cfg = config.paths.as_ref().ok_or_else(|| {
        CliError::Config("missing `paths` block (required for train)".to_string())
    })?;
    let data_path = paths_cfg.data.as_deref().ok_or_else(|| {
        CliError::Config("missing `paths.data` (required for train)".to_string())
    })?;
    let model_path = paths_cfg.model.as_deref().ok_or_else(|| {
        CliError::Config("missing `paths.model` (required for train)".to_string())
    })?;

    // Create network
    let mut ann = Ann::new(
        net_cfg.inputs,
        net_cfg.hidden_layers,
        net_cfg.hidden,
        net_cfg.outputs,
    )?;
    ann.activation_hidden = net_cfg.activation_hidden.into();
    ann.activation_output = net_cfg.activation_output.into();

    // Load data
    let data = parse_csv(data_path, net_cfg.inputs, net_cfg.outputs)?;
    eprintln!(
        "Loaded {} samples from {}",
        data.len(),
        data_path
    );

    let epochs = train_cfg.epochs;
    let lr = train_cfg.learning_rate;
    let progress_every = epochs.max(5) / 5;

    // Training loop
    if train_cfg.shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        for epoch in 0..epochs {
            indices.shuffle(&mut rng);
            for &i in &indices {
                ann.train(&data[i].0, &data[i].1, lr);
            }
            if (epoch + 1) % progress_every == 0 || epoch + 1 == epochs {
                eprintln!("Epoch {}/{epochs}", epoch + 1);
            }
        }
    } else {
        for epoch in 0..epochs {
            for (inp, tgt) in &data {
                ann.train(inp, tgt, lr);
            }
            if (epoch + 1) % progress_every == 0 || epoch + 1 == epochs {
                eprintln!("Epoch {}/{epochs}", epoch + 1);
            }
        }
    }

    // Save model
    let file = File::create(model_path)?;
    ann.write(file)?;
    eprintln!("Model saved to {model_path}");

    Ok(())
}

/// Run inference using a saved model.
pub fn run_inference(
    config: &Config,
    input_str: Option<&str>,
    data_override: Option<&str>,
    format: OutputFormat,
    stdout: &mut dyn Write,
) -> Result<(), CliError> {
    let paths_cfg = config.paths.as_ref().ok_or_else(|| {
        CliError::Config("missing `paths` block".to_string())
    })?;
    let model_path = paths_cfg.model.as_deref().ok_or_else(|| {
        CliError::Config("missing `paths.model`".to_string())
    })?;

    // Load model
    let file = File::open(model_path)?;
    let mut ann = Ann::read(BufReader::new(file))?;

    // Restore activations from config if network block is present
    if let Some(net_cfg) = &config.network {
        ann.activation_hidden = net_cfg.activation_hidden.into();
        ann.activation_output = net_cfg.activation_output.into();
    } else {
        eprintln!("note: no `network` block in config; activations default to sigmoid_cached");
    }

    if let Some(s) = input_str {
        // Single-sample inference
        let input = parse_input_str(s, ann.inputs)?;
        let output = ann.run(&input);
        writeln!(stdout, "{}", format_output(output, format))?;
    } else {
        // Batch inference
        let data_path = data_override
            .or(paths_cfg.data.as_deref())
            .ok_or_else(|| CliError::Config("no data source: provide --data or paths.data in config".to_string()))?;

        let file = File::open(data_path)?;
        let inputs = parse_csv_inputs(BufReader::new(file), ann.inputs)?;
        for input in &inputs {
            let output = ann.run(input);
            writeln!(stdout, "{}", format_output(output, format))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_csv_valid() {
        let data = "1.0,2.0,0.0,1.0\n3.0,4.0,1.0,0.0\n";
        let rows = parse_csv_reader(Cursor::new(data), 2, 2).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, vec![1.0, 2.0]);
        assert_eq!(rows[0].1, vec![0.0, 1.0]);
        assert_eq!(rows[1].0, vec![3.0, 4.0]);
        assert_eq!(rows[1].1, vec![1.0, 0.0]);
    }

    #[test]
    fn test_parse_csv_comment_skipping() {
        let data = "# header\n1.0,0.0,1.0\n\n# another comment\n0.0,1.0,0.0\n";
        let rows = parse_csv_reader(Cursor::new(data), 2, 1).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, vec![1.0, 0.0]);
        assert_eq!(rows[0].1, vec![1.0]);
    }

    #[test]
    fn test_parse_csv_wrong_column_count() {
        let data = "1.0,2.0\n";
        let result = parse_csv_reader(Cursor::new(data), 2, 2);
        assert!(matches!(result, Err(CliError::Csv { .. })));
    }

    #[test]
    fn test_parse_input_str_valid() {
        let vals = parse_input_str("1.0,2.0,3.0", 3).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_input_str_wrong_count() {
        let result = parse_input_str("1.0,2.0", 3);
        assert!(matches!(result, Err(CliError::Config(_))));
    }

    #[test]
    fn test_format_output_plain() {
        let s = format_output(&[0.023, 0.951, 0.025], OutputFormat::Plain);
        assert_eq!(s, "0.023 0.951 0.025");
    }

    #[test]
    fn test_format_output_json() {
        let s = format_output(&[0.023, 0.951, 0.025], OutputFormat::Json);
        assert_eq!(s, "[0.023, 0.951, 0.025]");
    }
}
