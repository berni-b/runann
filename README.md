# runann

A minimal, dependency-light feed-forward neural network library — a pure-Rust port of [genann](https://github.com/codeplea/genann).

Also ships a `runann` CLI binary for training and inference driven by a YAML config file.

---

## Quick Start

### Library

```toml
[dependencies]
runann = { path = "." }
```

```rust
use runann::{Activation, Ann};

// 2 inputs → 1 hidden layer of 3 neurons → 1 output
let mut ann = Ann::new(2, 1, 3, 1).unwrap();

// Training data: XOR gate
let inputs  = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
let targets = [[0.0],      [1.0],       [1.0],       [0.0]];

for _ in 0..5_000 {
    for (inp, tgt) in inputs.iter().zip(targets.iter()) {
        ann.train(inp, tgt, 0.5);
    }
}

for (inp, tgt) in inputs.iter().zip(targets.iter()) {
    let out = ann.run(inp);
    println!("XOR({}, {}) ≈ {:.3}  (expected {})", inp[0], inp[1], out[0], tgt[0]);
}
```

### CLI

```bash
# Build
cargo build --release

# Train an iris classifier
cargo run -- train --config examples/iris.yaml

# Batch inference (one output line per CSV row)
cargo run -- run --config examples/iris.yaml

# Single-sample inference
cargo run -- run --config examples/iris.yaml --input "5.1,3.5,1.4,0.2"

# JSON output format
cargo run -- run --config examples/iris.yaml --input "5.1,3.5,1.4,0.2" --format json

# Help
cargo run -- --help
cargo run -- train --help
cargo run -- run --help
```

---

## YAML Config Reference

```yaml
network:                        # Required for train; optional for run (restores activations)
  inputs:        4              # Number of input features (required)
  hidden_layers: 1              # Number of hidden layers (default: 0)
  hidden:        8              # Neurons per hidden layer (default: 0)
  outputs:       3              # Number of output neurons (required)
  activation_hidden: sigmoid_cached   # Activation for hidden layers (default: sigmoid_cached)
  activation_output: sigmoid_cached   # Activation for output layer (default: sigmoid_cached)

training:                       # Required for train
  epochs:        5000           # Number of training epochs (required)
  learning_rate: 0.5            # Backpropagation learning rate (required)
  shuffle:       false          # Shuffle training data each epoch (default: false)

paths:
  data:  examples/iris.data     # CSV data file (required for train; used for batch run)
  model: iris.ann               # Model file — train writes, run reads (required)
```

### Activation Functions

| Value            | Description                                      |
|------------------|--------------------------------------------------|
| `sigmoid_cached` | Fast sigmoid via 4096-entry lookup table (default) |
| `sigmoid`        | Exact sigmoid `1 / (1 + e^-a)`                  |
| `linear`         | Identity — output equals weighted sum            |
| `threshold`      | Step: 1.0 if a > 0, else 0.0                    |
| `relu`           | Rectified linear unit: max(0, a)                |

---

## CLI Subcommand Reference

### `runann train`

Trains a new network from a CSV file and saves the model.

```
runann [--config <FILE>] train
```

Requires `network`, `training`, and `paths` (with both `data` and `model`) in the config.

### `runann run`

Loads a saved model and runs inference.

```
runann [--config <FILE>] run [OPTIONS]

Options:
  --input <STR>     Single sample as "v1,v2,..."  (conflicts with --data)
  --data  <FILE>    Batch CSV file (overrides paths.data in config)
  --format          plain (default) | json
```

**Input resolution**: `--input` > `--data` > `config.paths.data`

If the `network` block is present in the config, activation functions are restored after loading the model. Otherwise a note is printed to stderr and activations default to `sigmoid_cached`.

---

## CSV Format

One sample per line. Columns: `input1,...,inputN,output1,...,outputM`.

Lines starting with `#` and empty lines are skipped.

For batch `run` inference, only the first `N` columns (inputs) are used; extra columns are allowed.

**Example** (XOR, 2 inputs + 1 output):

```csv
# XOR dataset
0.0,0.0,0.0
0.0,1.0,1.0
1.0,0.0,1.0
1.0,1.0,0.0
```

---

## Output Format

Progress and informational messages go to **stderr** so stdout stays pipeable.

```
# plain (default) — space-separated values, one line per sample
0.023 0.951 0.025

# json — array per sample
[0.023, 0.951, 0.025]
```

---

## Library API

### Creating a network

```rust
use runann::{Activation, Ann};

// inputs, hidden_layers, hidden_neurons_per_layer, outputs
let mut ann = Ann::new(4, 1, 8, 3).unwrap();

// Change activation functions (default: SigmoidCached)
ann.activation_hidden = Activation::Relu;
ann.activation_output = Activation::Sigmoid;
```

### Training

```rust
// Online backpropagation — one sample at a time
ann.train(&input_slice, &target_slice, learning_rate);
```

### Inference

```rust
let outputs: &[f64] = ann.run(&input_slice);
```

### Save / Load

```rust
use std::fs::File;
use std::io::BufReader;
use runann::Ann;

// Save
let file = File::create("model.ann").unwrap();
ann.write(file).unwrap();

// Load
let file = File::open("model.ann").unwrap();
let mut ann = Ann::read(BufReader::new(file)).unwrap();
// Restore activations if not using the default SigmoidCached
ann.activation_hidden = Activation::Relu;
```

The wire format is identical to genann's text format: a single space-separated line
`inputs hidden_layers hidden outputs w0 w1 … wN`.

---

## Weight Layout

Weights are stored in a flat `Vec<f64>`, layer by layer, neuron by neuron.
For each neuron the **bias weight comes first**, followed by one weight per
incoming connection:

```text
[ bias₀ | w₀₀ w₀₁ … | bias₁ | w₁₀ w₁₁ … | … ]
```

During the forward pass the bias contributes as `−bias_weight`, matching the genann convention.

---

## License

Zlib
