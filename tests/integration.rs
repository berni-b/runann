use runann::{Activation, Ann};
use std::io::Cursor;

/// Helper: assert floats are close within tolerance.
fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    assert!((a - b).abs() < tol, "{msg}: expected {b} ± {tol}, got {a}");
}

/// Mirrors test `basic`: single-neuron linear network with manually set weights.
#[test]
fn basic() {
    let mut ann = Ann::new(2, 0, 0, 1).unwrap();
    ann.activation_output = Activation::Linear;

    // Manually set weights: bias = 0.0, w0 = 1.0, w1 = 1.0
    {
        let w = ann.weights_mut();
        w[0] = 0.0; // bias
        w[1] = 1.0;
        w[2] = 1.0;
    }

    let out = ann.run(&[1.0, 1.0]);
    assert_close(out[0], 2.0, 1e-10, "basic 1+1");

    let out = ann.run(&[0.0, 1.0]);
    assert_close(out[0], 1.0, 1e-10, "basic 0+1");

    let out = ann.run(&[-1.0, 1.0]);
    assert_close(out[0], 0.0, 1e-10, "basic -1+1");
}

/// Mirrors test `xor`: manually wired XOR network with Threshold activation.
#[test]
fn xor() {
    // Architecture: 2 inputs, 1 hidden layer with 2 neurons, 1 output.
    // Hidden activation: Threshold; output activation: Threshold.
    let mut ann = Ann::new(2, 1, 2, 1).unwrap();
    ann.activation_hidden = Activation::Threshold;
    ann.activation_output = Activation::Threshold;

    // Weight layout per neuron: [bias_w, in0_w, in1_w, ...]
    // Forward pass applies bias as: sum = bias_w * -1.0 + Σ in_w * x
    // Threshold fires when sum > 0.
    //
    // XOR:  (i0 OR i1) AND NOT (i0 AND i1)
    // h0 = OR:  sum = -b + i0 + i1 > 0 for (i0+i1)≥1 only → b ∈ (0,1), use b=0.5
    // h1 = AND: sum = -b + i0 + i1 > 0 for (i0+i1)=2 only → b ∈ (1,2), use b=1.5
    // out = XOR: sum = -b + h0 - h1 > 0 when h0=1,h1=0 → b=0.5 works
    let w = ann.weights_mut();
    // hidden neuron 0 (OR)
    w[0] = 0.5; // bias_w → sum = -0.5 + i0 + i1; fires when i0+i1 ≥ 1
    w[1] = 1.0;
    w[2] = 1.0;
    // hidden neuron 1 (AND)
    w[3] = 1.5; // bias_w → sum = -1.5 + i0 + i1; fires when i0+i1 = 2
    w[4] = 1.0;
    w[5] = 1.0;
    // output neuron (XOR = OR AND NOT AND)
    w[6] = 0.5; // bias_w → sum = -0.5 + h0 - h1
    w[7] = 1.0;
    w[8] = -1.0;

    let inputs: &[[f64; 2]] = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected = [0.0, 1.0, 1.0, 0.0];

    for (inp, &exp) in inputs.iter().zip(expected.iter()) {
        let out = ann.run(inp.as_slice());
        assert_close(out[0], exp, 1e-10, &format!("xor({},{})", inp[0], inp[1]));
    }
}

/// Mirrors test `backprop`: error should decrease after 1000 training steps.
#[test]
fn backprop() {
    let mut ann = Ann::new(2, 1, 2, 1).unwrap();

    let inputs: &[[f64; 2]] = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0, 1.0, 1.0, 0.0];

    let error_before: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(inp, &t)| {
            let out = ann.run(inp.as_slice())[0];
            (out - t).powi(2)
        })
        .sum();

    for _ in 0..1000 {
        for (inp, &t) in inputs.iter().zip(targets.iter()) {
            ann.train(inp.as_slice(), &[t], 0.5);
        }
    }

    let error_after: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(inp, &t)| {
            let out = ann.run(inp.as_slice())[0];
            (out - t).powi(2)
        })
        .sum();

    assert!(
        error_after < error_before,
        "error should decrease: before={error_before}, after={error_after}"
    );
}

/// Learns the AND gate.
#[test]
fn train_and() {
    let mut ann = Ann::new(2, 1, 2, 1).unwrap();

    let inputs: &[[f64; 2]] = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0, 0.0, 0.0, 1.0];

    for _ in 0..5000 {
        for (inp, &t) in inputs.iter().zip(targets.iter()) {
            ann.train(inp.as_slice(), &[t], 0.5);
        }
    }

    for (inp, &t) in inputs.iter().zip(targets.iter()) {
        let out = ann.run(inp.as_slice())[0];
        assert_close(out, t, 0.1, &format!("AND({},{})", inp[0], inp[1]));
    }
}

/// Learns the OR gate.
#[test]
fn train_or() {
    let mut ann = Ann::new(2, 1, 2, 1).unwrap();

    let inputs: &[[f64; 2]] = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0, 1.0, 1.0, 1.0];

    for _ in 0..5000 {
        for (inp, &t) in inputs.iter().zip(targets.iter()) {
            ann.train(inp.as_slice(), &[t], 0.5);
        }
    }

    for (inp, &t) in inputs.iter().zip(targets.iter()) {
        let out = ann.run(inp.as_slice())[0];
        assert_close(out, t, 0.1, &format!("OR({},{})", inp[0], inp[1]));
    }
}

/// Learns the XOR gate (non-linearly separable, needs hidden layer).
#[test]
fn train_xor() {
    let mut ann = Ann::new(2, 1, 3, 1).unwrap();

    let inputs: &[[f64; 2]] = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0, 1.0, 1.0, 0.0];

    for _ in 0..10_000 {
        for (inp, &t) in inputs.iter().zip(targets.iter()) {
            ann.train(inp.as_slice(), &[t], 0.5);
        }
    }

    for (inp, &t) in inputs.iter().zip(targets.iter()) {
        let out = ann.run(inp.as_slice())[0];
        assert_close(out, t, 0.1, &format!("XOR({},{})", inp[0], inp[1]));
    }
}

/// Round-trips weights through write() + read().
#[test]
fn persist() {
    let mut ann = Ann::new(2, 1, 2, 1).unwrap();
    // Run once to set outputs_buf (shouldn't matter for persist, but mirrors C test).
    ann.run(&[0.5, 0.5]);

    let mut buf = Vec::new();
    ann.write(&mut buf).unwrap();

    let loaded = Ann::read(Cursor::new(&buf)).unwrap();

    assert_eq!(ann.inputs, loaded.inputs);
    assert_eq!(ann.hidden_layers, loaded.hidden_layers);
    assert_eq!(ann.hidden, loaded.hidden);
    assert_eq!(ann.outputs, loaded.outputs);

    for (a, b) in ann.weights_ref().iter().zip(loaded.weights_ref().iter()) {
        assert_close(*a, *b, 1e-15, "weight round-trip");
    }
}

/// clone() produces an independent copy.
#[test]
fn copy() {
    let ann = Ann::new(2, 1, 2, 1).unwrap();
    let mut copy = ann.clone();

    // Mutate the copy's weights.
    for w in copy.weights_mut() {
        *w += 100.0;
    }

    // Original should be unchanged.
    for (a, b) in ann.weights_ref().iter().zip(copy.weights_ref().iter()) {
        assert!(
            (a - b).abs() > 50.0,
            "original should differ from mutated copy"
        );
    }
}

/// Cached sigmoid agrees with exact sigmoid within 2e-3 over [-15, 15].
///
/// A 4096-entry table over a 30-unit domain has step ≈ 0.0073.
/// At worst (maximum sigmoid slope of 0.25) the error is ≈ 0.0025.
#[test]
fn sigmoid_approx() {
    for i in 0..=300 {
        let x = -15.0 + (i as f64) / 10.0;
        let cached = Activation::SigmoidCached.apply(x);
        let exact = Activation::Sigmoid.apply(x);
        assert_close(cached, exact, 2e-3, &format!("sigmoid({x})"));
    }
}
