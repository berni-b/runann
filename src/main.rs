//! `runann` CLI — train and run feed-forward neural networks from YAML config.

use std::fs;
use std::io;
use std::process;

use clap::{Parser, Subcommand};

use runann::cli::commands::{run_inference, run_train, CliError, OutputFormat};
use runann::cli::config::Config;

#[derive(Parser)]
#[command(
    name = "runann",
    about = "Train and run feed-forward neural networks",
    version
)]
struct Cli {
    /// YAML configuration file
    #[arg(short, long, default_value = "runann.yaml")]
    config: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a network from CSV data and save it
    Train,

    /// Load a saved network and run inference
    Run {
        /// Single input sample as "v1,v2,..." (conflicts with --data)
        #[arg(long, conflicts_with = "data")]
        input: Option<String>,

        /// Batch CSV file (overrides paths.data in config)
        #[arg(long)]
        data: Option<String>,

        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let yaml = match fs::read_to_string(&cli.config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read config file '{}': {e}", cli.config);
            process::exit(1);
        }
    };

    let config = match Config::from_yaml(&yaml) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", CliError::Yaml(e));
            process::exit(1);
        }
    };

    let result = match &cli.command {
        Command::Train => run_train(&config),
        Command::Run { input, data, format } => {
            let fmt = match format.as_str() {
                "json" => OutputFormat::Json,
                "plain" => OutputFormat::Plain,
                other => {
                    eprintln!("error: unknown format '{other}'; use 'plain' or 'json'");
                    process::exit(1);
                }
            };
            run_inference(
                &config,
                input.as_deref(),
                data.as_deref(),
                fmt,
                &mut io::stdout(),
            )
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
