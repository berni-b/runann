//! Forward (inference) pass.

use crate::ann::Ann;

impl Ann {
    /// Run the network forward on `inputs` and return a view of the output neurons.
    ///
    /// The returned slice has length [`self.outputs`](Ann::outputs) and is
    /// backed by an internal buffer that is overwritten on the next call to
    /// `run` or [`train`](Ann::train).
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len() != self.inputs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use runann::Ann;
    ///
    /// let mut ann = Ann::new(2, 1, 4, 1).unwrap();
    /// let out = ann.run(&[0.0, 1.0]);
    /// assert_eq!(out.len(), 1);
    /// ```
    pub fn run(&mut self, inputs: &[f64]) -> &[f64] {
        // Copy inputs into the echo zone.
        self.outputs_buf[..self.inputs].copy_from_slice(inputs);

        let mut weight_idx = 0usize;
        let mut neuron_idx = self.inputs; // first hidden/output neuron in outputs_buf

        // Hidden layers
        for layer in 0..self.hidden_layers {
            let n_in = if layer == 0 { self.inputs } else { self.hidden };
            let in_start = if layer == 0 {
                0
            } else {
                self.inputs + (layer - 1) * self.hidden
            };
            let activation = self.activation_hidden;

            for _ in 0..self.hidden {
                let mut sum = -self.weights[weight_idx]; // bias
                for i in 0..n_in {
                    sum += self.weights[weight_idx + 1 + i] * self.outputs_buf[in_start + i];
                }
                self.outputs_buf[neuron_idx] = activation.apply(sum);
                weight_idx += n_in + 1;
                neuron_idx += 1;
            }
        }

        // Output layer
        let n_in = if self.hidden_layers == 0 {
            self.inputs
        } else {
            self.hidden
        };
        let in_start = if self.hidden_layers == 0 {
            0
        } else {
            self.inputs + (self.hidden_layers - 1) * self.hidden
        };
        let activation = self.activation_output;

        for _ in 0..self.outputs {
            let mut sum = -self.weights[weight_idx]; // bias
            for i in 0..n_in {
                sum += self.weights[weight_idx + 1 + i] * self.outputs_buf[in_start + i];
            }
            self.outputs_buf[neuron_idx] = activation.apply(sum);
            weight_idx += n_in + 1;
            neuron_idx += 1;
        }

        // Return the output neurons slice.
        let out_start = self.inputs + self.hidden_layers * self.hidden;
        &self.outputs_buf[out_start..out_start + self.outputs]
    }
}
