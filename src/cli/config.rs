//! Serde-deserializable config structs for the `runann` CLI.

use serde::Deserialize;

use crate::Activation;

/// Top-level configuration file structure.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Network topology and activation functions (required for `train`).
    pub network: Option<NetworkConfig>,
    /// Training hyperparameters (required for `train`).
    pub training: Option<TrainingConfig>,
    /// File paths for data and model.
    pub paths: Option<PathsConfig>,
}

/// Network topology and activation configuration.
#[derive(Debug, Deserialize)]
pub struct NetworkConfig {
    /// Number of input features.
    pub inputs: usize,
    /// Number of hidden layers (0 for single-layer perceptron).
    #[serde(default)]
    pub hidden_layers: usize,
    /// Number of neurons per hidden layer.
    #[serde(default)]
    pub hidden: usize,
    /// Number of output neurons.
    pub outputs: usize,
    /// Activation function for hidden layers.
    #[serde(default)]
    pub activation_hidden: ActivationConfig,
    /// Activation function for output layer.
    #[serde(default)]
    pub activation_output: ActivationConfig,
}

/// Training hyperparameters.
#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for backpropagation.
    pub learning_rate: f64,
    /// Whether to shuffle training data each epoch.
    #[serde(default)]
    pub shuffle: bool,
}

/// File path configuration.
#[derive(Debug, Deserialize, Default)]
pub struct PathsConfig {
    /// Path to CSV training/inference data file.
    pub data: Option<String>,
    /// Path to save/load the model file.
    pub model: Option<String>,
}

/// Activation function selection in config.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActivationConfig {
    #[default]
    SigmoidCached,
    Sigmoid,
    Linear,
    Threshold,
    Relu,
}

impl From<ActivationConfig> for Activation {
    fn from(a: ActivationConfig) -> Self {
        match a {
            ActivationConfig::SigmoidCached => Activation::SigmoidCached,
            ActivationConfig::Sigmoid => Activation::Sigmoid,
            ActivationConfig::Linear => Activation::Linear,
            ActivationConfig::Threshold => Activation::Threshold,
            ActivationConfig::Relu => Activation::Relu,
        }
    }
}

impl Config {
    /// Parse a YAML string into a [`Config`].
    pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_yaml_deserialization() {
        let yaml = r#"
network:
  inputs: 4
  hidden_layers: 1
  hidden: 8
  outputs: 3
  activation_hidden: sigmoid_cached
  activation_output: sigmoid_cached

training:
  epochs: 5000
  learning_rate: 0.5
  shuffle: false

paths:
  data: examples/iris.data
  model: iris.ann
"#;
        let cfg = Config::from_yaml(yaml).unwrap();
        let net = cfg.network.unwrap();
        assert_eq!(net.inputs, 4);
        assert_eq!(net.hidden_layers, 1);
        assert_eq!(net.hidden, 8);
        assert_eq!(net.outputs, 3);
        assert_eq!(net.activation_hidden, ActivationConfig::SigmoidCached);
        assert_eq!(net.activation_output, ActivationConfig::SigmoidCached);

        let tr = cfg.training.unwrap();
        assert_eq!(tr.epochs, 5000);
        assert!((tr.learning_rate - 0.5).abs() < 1e-10);
        assert!(!tr.shuffle);

        let paths = cfg.paths.unwrap();
        assert_eq!(paths.data.as_deref(), Some("examples/iris.data"));
        assert_eq!(paths.model.as_deref(), Some("iris.ann"));
    }

    #[test]
    fn test_activation_variants() {
        for (s, expected) in [
            ("sigmoid_cached", ActivationConfig::SigmoidCached),
            ("sigmoid", ActivationConfig::Sigmoid),
            ("linear", ActivationConfig::Linear),
            ("threshold", ActivationConfig::Threshold),
            ("relu", ActivationConfig::Relu),
        ] {
            let yaml = format!("network:\n  inputs: 2\n  outputs: 1\n  activation_hidden: {s}\n  activation_output: {s}");
            let cfg = Config::from_yaml(&yaml).unwrap();
            let net = cfg.network.unwrap();
            assert_eq!(net.activation_hidden, expected, "activation_hidden for {s}");
            assert_eq!(net.activation_output, expected, "activation_output for {s}");
        }
    }

    #[test]
    fn test_default_values() {
        let yaml = "network:\n  inputs: 2\n  outputs: 1\n";
        let cfg = Config::from_yaml(yaml).unwrap();
        let net = cfg.network.unwrap();
        assert_eq!(net.hidden_layers, 0);
        assert_eq!(net.hidden, 0);
        assert_eq!(net.activation_hidden, ActivationConfig::SigmoidCached);
        assert_eq!(net.activation_output, ActivationConfig::SigmoidCached);
    }

    #[test]
    fn test_shuffle_default_false() {
        let yaml = "training:\n  epochs: 100\n  learning_rate: 0.1\n";
        let cfg = Config::from_yaml(yaml).unwrap();
        let tr = cfg.training.unwrap();
        assert!(!tr.shuffle);
    }

    #[test]
    fn test_invalid_activation_rejected() {
        let yaml = "network:\n  inputs: 2\n  outputs: 1\n  activation_hidden: foobar\n";
        assert!(Config::from_yaml(yaml).is_err());
    }

    #[test]
    fn test_activation_conversion() {
        assert_eq!(Activation::from(ActivationConfig::SigmoidCached), Activation::SigmoidCached);
        assert_eq!(Activation::from(ActivationConfig::Sigmoid), Activation::Sigmoid);
        assert_eq!(Activation::from(ActivationConfig::Linear), Activation::Linear);
        assert_eq!(Activation::from(ActivationConfig::Threshold), Activation::Threshold);
        assert_eq!(Activation::from(ActivationConfig::Relu), Activation::Relu);
    }
}
