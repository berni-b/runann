//! Port of genann example2.c — XOR learned by random restart (hill climbing).
use runann::Ann;

fn total_error(ann: &mut Ann) -> f64 {
    let input = [[0.0f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let output = [0.0f64, 1.0, 1.0, 0.0];

    let mut error = 0.0;
    for i in 0..4 {
        let out = ann.run(&input[i]);
        error += (out[0] - output[i]).powi(2);
    }
    error
}

fn main() {
    let input = [[0.0f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let output = [[0.0f64], [1.0], [1.0], [0.0]];

    let mut best = Ann::new(2, 1, 2, 1).expect("failed to create network");
    let mut best_error = total_error(&mut best);

    println!("Starting error: {best_error:.6}");

    for _ in 0..50 {
        let mut candidate = Ann::new(2, 1, 2, 1).expect("failed to create network");
        candidate.randomize();

        // Train candidate with backprop briefly.
        for _ in 0..500 {
            for i in 0..4 {
                candidate.train(&input[i], &output[i], 3.0);
            }
        }

        let err = total_error(&mut candidate);
        if err < best_error {
            best_error = err;
            best = candidate;
        }
    }

    println!("Best error: {best_error:.6}");
    println!("Results:");
    for i in 0..4 {
        let out = best.run(&input[i]);
        println!(
            "({}, {}) -> {:.6}",
            input[i][0] as i32, input[i][1] as i32, out[0]
        );
    }
}
