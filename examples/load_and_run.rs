//! Port of genann example3.c — save a trained network and reload it.
use runann::Ann;
use std::io::{BufReader, Cursor};

fn main() {
    let input = [[0.0f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let output = [[0.0f64], [1.0], [1.0], [0.0]];

    // Train.
    let mut ann = Ann::new(2, 1, 2, 1).expect("failed to create network");
    for _ in 0..300 {
        for i in 0..4 {
            ann.train(&input[i], &output[i], 3.0);
        }
    }

    // Serialize to an in-memory buffer.
    let mut buf: Vec<u8> = Vec::new();
    ann.write(&mut buf).expect("write failed");

    println!("Serialized network ({} bytes):", buf.len());
    println!("{}", String::from_utf8_lossy(&buf).trim());

    // Reload from buffer.
    let mut loaded = Ann::read(BufReader::new(Cursor::new(&buf))).expect("read failed");

    println!("\nResults after reload:");
    for i in 0..4 {
        let out = loaded.run(&input[i]);
        println!(
            "({}, {}) -> {:.6}",
            input[i][0] as i32, input[i][1] as i32, out[0]
        );
    }
}
