//! Port of genann example1.c — XOR learned via backpropagation.
use runann::{Activation, Ann};

fn main() {
    let input = [[0.0f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let output = [[0.0f64], [1.0], [1.0], [0.0]];

    let mut ann = Ann::new(2, 1, 2, 1).expect("failed to create network");
    // The default activation is SigmoidCached, same as genann default.
    // Override hidden activation to keep it explicit.
    ann.activation_hidden = Activation::SigmoidCached;
    ann.activation_output = Activation::SigmoidCached;

    println!("Training for 300 iterations ...");
    for _ in 0..300 {
        for i in 0..4 {
            ann.train(&input[i], &output[i], 3.0);
        }
    }

    println!("Trained. Results:");
    for i in 0..4 {
        let out = ann.run(&input[i]);
        println!(
            "({}, {}) -> {:.6}",
            input[i][0] as i32, input[i][1] as i32, out[0]
        );
    }
}
