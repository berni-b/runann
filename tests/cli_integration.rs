//! Integration tests for the `runann` CLI commands.

use std::fs;
use std::io::Cursor;

use runann::cli::commands::{run_inference, run_train, CliError, OutputFormat};
use runann::cli::config::Config;

fn xor_csv() -> &'static str {
    "0.0,0.0,0.0\n\
     0.0,1.0,1.0\n\
     1.0,0.0,1.0\n\
     1.0,1.0,0.0\n"
}

fn xor_config_yaml(data_path: &str, model_path: &str) -> String {
    format!(
        "network:\n  inputs: 2\n  hidden_layers: 1\n  hidden: 4\n  outputs: 1\n\
         training:\n  epochs: 2000\n  learning_rate: 0.5\n\
         paths:\n  data: {data_path}\n  model: {model_path}\n"
    )
}

fn model_only_config_yaml(model_path: &str) -> String {
    format!(
        "network:\n  inputs: 2\n  hidden_layers: 1\n  hidden: 4\n  outputs: 1\n\
         paths:\n  model: {model_path}\n"
    )
}

#[test]
fn train_xor() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("xor.csv");
    let model_path = dir.path().join("xor.ann");

    fs::write(&data_path, xor_csv()).unwrap();

    let yaml = xor_config_yaml(
        data_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();

    run_train(&config).unwrap();

    assert!(model_path.exists(), "model file should be created");
    let model_content = fs::read_to_string(&model_path).unwrap();
    assert!(!model_content.is_empty());
    // First token should be "2" (inputs)
    assert!(model_content.trim_start().starts_with("2 "));
}

#[test]
fn run_single_input() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("xor.csv");
    let model_path = dir.path().join("xor.ann");

    fs::write(&data_path, xor_csv()).unwrap();

    let yaml = xor_config_yaml(
        data_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    run_train(&config).unwrap();

    // Run single-sample inference
    let run_yaml = model_only_config_yaml(model_path.to_str().unwrap());
    let run_config = Config::from_yaml(&run_yaml).unwrap();

    let mut out = Vec::new();
    run_inference(
        &run_config,
        Some("0.0,0.0"),
        None,
        OutputFormat::Plain,
        &mut out,
    )
    .unwrap();

    let output = String::from_utf8(out).unwrap();
    let output = output.trim();
    // Should be one line with one value
    assert!(!output.is_empty(), "output should not be empty");
    let _: f64 = output.parse().expect("output should be a float");
}

#[test]
fn run_batch_csv() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("xor.csv");
    let model_path = dir.path().join("xor.ann");

    fs::write(&data_path, xor_csv()).unwrap();

    let yaml = xor_config_yaml(
        data_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    run_train(&config).unwrap();

    // Batch inference CSV (only input columns needed)
    let batch_csv = "0.0,0.0\n0.0,1.0\n1.0,0.0\n1.0,1.0\n";
    let batch_path = dir.path().join("batch.csv");
    fs::write(&batch_path, batch_csv).unwrap();

    let run_yaml = format!(
        "network:\n  inputs: 2\n  hidden_layers: 1\n  hidden: 4\n  outputs: 1\n\
         paths:\n  model: {}\n  data: {}\n",
        model_path.to_str().unwrap(),
        batch_path.to_str().unwrap(),
    );
    let run_config = Config::from_yaml(&run_yaml).unwrap();

    let mut out = Vec::new();
    run_inference(&run_config, None, None, OutputFormat::Plain, &mut out).unwrap();

    let output = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = output.trim().lines().collect();
    assert_eq!(lines.len(), 4, "should have one output line per input sample");
}

#[test]
fn run_json_format() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("xor.csv");
    let model_path = dir.path().join("xor.ann");

    fs::write(&data_path, xor_csv()).unwrap();

    let yaml = xor_config_yaml(
        data_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    run_train(&config).unwrap();

    let run_yaml = model_only_config_yaml(model_path.to_str().unwrap());
    let run_config = Config::from_yaml(&run_yaml).unwrap();

    let mut out = Vec::new();
    run_inference(
        &run_config,
        Some("1.0,0.0"),
        None,
        OutputFormat::Json,
        &mut out,
    )
    .unwrap();

    let output = String::from_utf8(out).unwrap();
    let output = output.trim();
    assert!(output.starts_with('[') && output.ends_with(']'), "JSON output should be array");
}

#[test]
fn train_missing_network_block() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("xor.csv");
    let model_path = dir.path().join("xor.ann");
    fs::write(&data_path, xor_csv()).unwrap();

    let yaml = format!(
        "training:\n  epochs: 100\n  learning_rate: 0.5\n\
         paths:\n  data: {}\n  model: {}\n",
        data_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();

    match run_train(&config) {
        Err(CliError::Config(msg)) => assert!(msg.contains("network")),
        other => panic!("expected CliError::Config, got {other:?}"),
    }
}

#[test]
fn train_missing_paths_data() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("xor.ann");

    let yaml = format!(
        "network:\n  inputs: 2\n  hidden_layers: 1\n  hidden: 4\n  outputs: 1\n\
         training:\n  epochs: 100\n  learning_rate: 0.5\n\
         paths:\n  model: {}\n",
        model_path.to_str().unwrap(),
    );
    let config = Config::from_yaml(&yaml).unwrap();

    match run_train(&config) {
        Err(CliError::Config(msg)) => assert!(msg.contains("paths.data")),
        other => panic!("expected CliError::Config, got {other:?}"),
    }
}

#[test]
fn csv_bad_column_count() {
    use runann::cli::commands::parse_csv_reader;

    // 2 inputs + 1 output expected but only 2 columns provided
    let data = "1.0,0.0\n";
    let result = parse_csv_reader(Cursor::new(data), 2, 1);
    match result {
        Err(CliError::Csv { line: 1, .. }) => {}
        other => panic!("expected CliError::Csv at line 1, got {other:?}"),
    }
}
