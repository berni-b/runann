//! Online backpropagation training.

use crate::ann::Ann;

impl Ann {
    /// Train the network on a single example using backpropagation.
    ///
    /// Performs one forward pass followed by one gradient-descent weight update.
    /// Call this in a loop over your training set; for best results shuffle the
    /// data between epochs.
    ///
    /// # Arguments
    ///
    /// - `inputs` — feature vector; must have length [`self.inputs`](Ann::inputs).
    /// - `desired` — target output vector; must have length
    ///   [`self.outputs`](Ann::outputs).
    /// - `learning_rate` — step size (e.g. `0.5`).  Larger values converge
    ///   faster but may overshoot; smaller values are more stable.
    ///
    /// # Example
    ///
    /// ```rust
    /// use runann::Ann;
    ///
    /// let mut ann = Ann::new(2, 1, 4, 1).unwrap();
    ///
    /// // One gradient step on a single (AND gate) example:
    /// ann.train(&[1.0, 1.0], &[1.0], 0.5);
    /// ```
    pub fn train(&mut self, inputs: &[f64], desired: &[f64], learning_rate: f64) {
        // Step 1: forward pass (fills outputs_buf).
        self.run(inputs);

        let out_delta_start = self.hidden_layers * self.hidden;

        // Step 2: output layer deltas.
        for (j, &target) in desired.iter().enumerate().take(self.outputs) {
            let o = self.outputs_buf[self.inputs + out_delta_start + j];
            self.deltas[out_delta_start + j] = if self.activation_output.is_linear() {
                target - o
            } else {
                (target - o) * o * (1.0 - o)
            };
        }

        // Step 3: hidden layer deltas (reverse order).
        if self.hidden_layers > 0 {
            for l in (0..self.hidden_layers).rev() {
                // Determine where the next layer's deltas and weights start.
                let next_delta_start = (l + 1) * self.hidden; // 0-based delta index for next layer
                let (next_weight_offset, next_layer_size) = if l + 1 == self.hidden_layers {
                    // Next layer is the output layer.
                    // Weight offset for output layer:
                    let offset = if self.hidden_layers == 0 {
                        0
                    } else {
                        (self.inputs + 1) * self.hidden
                            + (self.hidden_layers - 1) * (self.hidden + 1) * self.hidden
                    };
                    (offset, self.outputs)
                } else {
                    // Next layer is a hidden layer (l+1).
                    // Weights for hidden layer l+1:
                    // first hidden layer: (inputs+1)*hidden weights
                    // layers 1..l+1: (hidden+1)*hidden each
                    let offset =
                        (self.inputs + 1) * self.hidden + l * (self.hidden + 1) * self.hidden;
                    (offset, self.hidden)
                };

                for j in 0..self.hidden {
                    let o = self.outputs_buf[self.inputs + l * self.hidden + j];
                    // Sum contributions from every neuron k in the next layer.
                    let mut sum = 0.0f64;
                    for k in 0..next_layer_size {
                        // Weight of input j+1 (skip bias at +0) for neuron k in next layer.
                        sum += self.deltas[next_delta_start + k]
                            * self.weights[next_weight_offset + k * (self.hidden + 1) + 1 + j];
                    }
                    self.deltas[l * self.hidden + j] = o * (1.0 - o) * sum;
                }
            }
        }

        // Step 4: weight updates (forward order).
        let mut weight_idx = 0usize;
        let mut delta_idx = 0usize;

        // Hidden layers.
        for layer in 0..self.hidden_layers {
            let n_in = if layer == 0 { self.inputs } else { self.hidden };
            let in_start = if layer == 0 {
                0
            } else {
                self.inputs + (layer - 1) * self.hidden
            };

            for j in 0..self.hidden {
                let d = self.deltas[delta_idx + j];
                // Bias weight update.
                self.weights[weight_idx] += -(d * learning_rate);
                // Input weight updates.
                for i in 0..n_in {
                    self.weights[weight_idx + 1 + i] +=
                        d * learning_rate * self.outputs_buf[in_start + i];
                }
                weight_idx += n_in + 1;
            }
            delta_idx += self.hidden;
        }

        // Output layer.
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

        for j in 0..self.outputs {
            let d = self.deltas[delta_idx + j];
            self.weights[weight_idx] += -(d * learning_rate);
            for i in 0..n_in {
                self.weights[weight_idx + 1 + i] +=
                    d * learning_rate * self.outputs_buf[in_start + i];
            }
            weight_idx += n_in + 1;
        }
    }
}
