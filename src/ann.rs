//! Core [`Ann`] struct and topology helpers.

use crate::activation::Activation;
use crate::error::{AnnError, Result};

/// Compute `(total_weights, total_neurons)` for the given topology.
pub(crate) fn compute_sizes(
    inputs: usize,
    hidden_layers: usize,
    hidden: usize,
    outputs: usize,
) -> (usize, usize) {
    let total_neurons = hidden_layers * hidden + outputs;
    let total_weights = if hidden_layers == 0 {
        (inputs + 1) * outputs
    } else {
        (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden + (hidden + 1) * outputs
    };
    (total_weights, total_neurons)
}

/// A fully-connected feed-forward neural network.
///
/// # Topology
///
/// The network has:
/// - `inputs` input neurons (no weights; the raw feature values)
/// - `hidden_layers` hidden layers, each with `hidden` neurons
/// - `outputs` output neurons
///
/// When `hidden_layers == 0` the network is a single-layer perceptron connecting
/// inputs directly to outputs.
///
/// # Activation functions
///
/// [`activation_hidden`](Ann::activation_hidden) is applied to every hidden
/// neuron; [`activation_output`](Ann::activation_output) is applied to every
/// output neuron.  Both default to [`Activation::SigmoidCached`].
///
/// # Cloning
///
/// `Ann` derives [`Clone`], which produces a fully independent copy (equivalent
/// to `genann_copy`).
///
/// # Example
///
/// ```rust
/// use runann::{Activation, Ann};
///
/// let mut ann = Ann::new(2, 1, 4, 1).unwrap();
/// ann.activation_output = Activation::Linear;
///
/// let out = ann.run(&[0.5, 0.5]);
/// println!("{}", out[0]);
/// ```
#[derive(Clone)]
pub struct Ann {
    /// Number of input features.
    pub inputs: usize,
    /// Number of hidden layers (0 for a single-layer perceptron).
    pub hidden_layers: usize,
    /// Number of neurons per hidden layer.
    pub hidden: usize,
    /// Number of output neurons.
    pub outputs: usize,
    /// Activation function used for all hidden neurons.
    pub activation_hidden: Activation,
    /// Activation function used for all output neurons.
    pub activation_output: Activation,
    #[allow(dead_code)]
    pub(crate) total_weights: usize,
    #[allow(dead_code)]
    pub(crate) total_neurons: usize,
    pub(crate) weights: Vec<f64>,
    pub(crate) outputs_buf: Vec<f64>,
    pub(crate) deltas: Vec<f64>,
}

impl Ann {
    /// Create a new network with random weights drawn uniformly from `[−0.5, 0.5]`.
    ///
    /// # Errors
    ///
    /// Returns [`AnnError::InvalidTopology`] if:
    /// - `inputs == 0`
    /// - `outputs == 0`
    /// - `hidden_layers > 0` but `hidden == 0`
    ///
    /// # Example
    ///
    /// ```rust
    /// use runann::Ann;
    ///
    /// // 3 inputs, 2 hidden layers of 5 neurons each, 2 outputs
    /// let ann = Ann::new(3, 2, 5, 2).unwrap();
    /// assert_eq!(ann.inputs, 3);
    /// assert_eq!(ann.outputs, 2);
    /// ```
    pub fn new(inputs: usize, hidden_layers: usize, hidden: usize, outputs: usize) -> Result<Self> {
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

        let mut ann = Ann {
            inputs,
            hidden_layers,
            hidden,
            outputs,
            activation_hidden: Activation::default(),
            activation_output: Activation::default(),
            total_weights,
            total_neurons,
            weights: vec![0.0; total_weights],
            outputs_buf: vec![0.0; inputs + total_neurons],
            deltas: vec![0.0; total_neurons],
        };
        ann.randomize();
        Ok(ann)
    }

    /// Re-initialize all weights with fresh uniform random values in `[−0.5, 0.5]`.
    ///
    /// Useful for random-restart training strategies.
    pub fn randomize(&mut self) {
        let mut rng = rand::rng();
        for w in &mut self.weights {
            *w = rand::Rng::random::<f64>(&mut rng) - 0.5;
        }
    }

    /// Read-only view of the flat weight buffer.
    ///
    /// The layout is layer-by-layer, neuron-by-neuron, with the bias weight
    /// first for each neuron.  See the [crate-level docs](crate) for details.
    pub fn weights_ref(&self) -> &[f64] {
        &self.weights
    }

    /// Mutable view of the flat weight buffer.
    ///
    /// Use this to set weights manually, e.g. when implementing a hand-crafted
    /// logic gate or loading weights from a custom format.
    ///
    /// # Example
    ///
    /// ```rust
    /// use runann::{Activation, Ann};
    ///
    /// // Single-neuron linear network: output = i0 + i1
    /// let mut ann = Ann::new(2, 0, 0, 1).unwrap();
    /// ann.activation_output = Activation::Linear;
    /// let w = ann.weights_mut();
    /// w[0] = 0.0; // bias
    /// w[1] = 1.0; // weight for input 0
    /// w[2] = 1.0; // weight for input 1
    ///
    /// assert!((ann.run(&[3.0, 4.0])[0] - 7.0).abs() < 1e-10);
    /// ```
    pub fn weights_mut(&mut self) -> &mut [f64] {
        &mut self.weights
    }
}
