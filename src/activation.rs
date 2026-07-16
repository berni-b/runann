//! Activation functions for hidden and output neurons.

use std::sync::OnceLock;

const TABLE_SIZE: usize = 4096;
const DOMAIN_MIN: f64 = -15.0;
const DOMAIN_MAX: f64 = 15.0;
const DOMAIN_RANGE: f64 = DOMAIN_MAX - DOMAIN_MIN; // 30.0

static SIGMOID_TABLE: OnceLock<[f64; TABLE_SIZE]> = OnceLock::new();

fn sigmoid_table() -> &'static [f64; TABLE_SIZE] {
    SIGMOID_TABLE.get_or_init(|| {
        let mut table = [0.0f64; TABLE_SIZE];
        for (i, slot) in table.iter_mut().enumerate() {
            let x = DOMAIN_MIN + (i as f64 / TABLE_SIZE as f64) * DOMAIN_RANGE;
            *slot = 1.0 / (1.0 + (-x).exp());
        }
        table
    })
}

fn sigmoid_exact(a: f64) -> f64 {
    if a < -45.0 {
        0.0
    } else if a > 45.0 {
        1.0
    } else {
        1.0 / (1.0 + (-a).exp())
    }
}

fn sigmoid_cached(a: f64) -> f64 {
    if a < DOMAIN_MIN {
        return 0.0;
    }
    if a >= DOMAIN_MAX {
        return 1.0;
    }
    let idx = ((a - DOMAIN_MIN) / DOMAIN_RANGE * TABLE_SIZE as f64) as usize;
    let idx = idx.min(TABLE_SIZE - 1);
    sigmoid_table()[idx]
}

/// Activation function applied to each neuron's weighted sum.
///
/// Set [`Ann::activation_hidden`](crate::Ann::activation_hidden) and
/// [`Ann::activation_output`](crate::Ann::activation_output) before training or
/// inference to choose the function used in each layer.
///
/// The default (and genann's default) is [`SigmoidCached`](Activation::SigmoidCached).
///
/// # Example
///
/// ```rust
/// use runann::{Activation, Ann};
///
/// let mut ann = Ann::new(2, 1, 4, 1).unwrap();
/// ann.activation_hidden = Activation::Relu;
/// ann.activation_output = Activation::Sigmoid;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Fast sigmoid via a 4 096-entry lookup table covering `[−15, 15]`.
    ///
    /// Values outside the domain are clamped to 0 or 1.  The approximation
    /// error relative to the exact sigmoid is at most ~2 × 10⁻³.
    ///
    /// This is the default and mirrors genann's behaviour.
    #[default]
    SigmoidCached,

    /// Exact sigmoid `1 / (1 + e^−a)`, clipped at ±45 to avoid overflow.
    Sigmoid,

    /// Identity function — output equals the weighted sum.
    ///
    /// Using `Linear` on the output layer gives a plain regression network.
    Linear,

    /// Step function: returns `1.0` if `a > 0`, otherwise `0.0`.
    Threshold,

    /// Rectified linear unit: `max(0, a)`.
    Relu,
}

impl Activation {
    /// Evaluate the activation function at `a`.
    ///
    /// ```rust
    /// use runann::Activation;
    ///
    /// assert_eq!(Activation::Linear.apply(3.7), 3.7);
    /// assert_eq!(Activation::Threshold.apply(-1.0), 0.0);
    /// assert_eq!(Activation::Threshold.apply(0.5), 1.0);
    /// assert_eq!(Activation::Relu.apply(-2.0), 0.0);
    /// assert_eq!(Activation::Relu.apply(2.0), 2.0);
    /// ```
    pub fn apply(self, a: f64) -> f64 {
        match self {
            Activation::SigmoidCached => sigmoid_cached(a),
            Activation::Sigmoid => sigmoid_exact(a),
            Activation::Linear => a,
            Activation::Threshold => {
                if a > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Relu => a.max(0.0),
        }
    }

    pub(crate) fn is_linear(self) -> bool {
        self == Activation::Linear
    }
}
