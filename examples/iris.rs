//! Port of genann example4.c — Iris dataset classification.
//!
//! Expects `examples/iris.data` in CSV format (UCI Iris dataset):
//!   sepal_length,sepal_width,petal_length,petal_width,class_label
//! where class_label is one of: Iris-setosa, Iris-versicolor, Iris-virginica
use runann::Ann;
use std::fs::File;
use std::io::{BufRead, BufReader};

const SAMPLES: usize = 150;
const INPUTS: usize = 4;
const OUTPUTS: usize = 3;

fn class_to_index(label: &str) -> Option<usize> {
    match label.trim() {
        "Iris-setosa" => Some(0),
        "Iris-versicolor" => Some(1),
        "Iris-virginica" => Some(2),
        _ => None,
    }
}

fn main() {
    let path = "examples/iris.data";
    let file = File::open(path).unwrap_or_else(|_| {
        eprintln!("Cannot open {path}. Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
        std::process::exit(1);
    });

    let mut input_data = vec![[0.0f64; INPUTS]; SAMPLES];
    let mut target_data = vec![[0.0f64; OUTPUTS]; SAMPLES];

    let mut count = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line.expect("read error");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            continue;
        }
        for j in 0..INPUTS {
            input_data[count][j] = parts[j].parse().expect("bad float");
        }
        let cls = class_to_index(parts[4]).unwrap_or_else(|| panic!("unknown class: {}", parts[4]));
        target_data[count][cls] = 1.0;

        count += 1;
        if count >= SAMPLES {
            break;
        }
    }
    assert_eq!(count, SAMPLES, "expected {SAMPLES} samples");

    let mut ann = Ann::new(INPUTS, 1, 4, OUTPUTS).expect("failed to create network");

    println!("Training on {count} iris samples ...");
    for _ in 0..5000 {
        for i in 0..count {
            ann.train(&input_data[i], &target_data[i], 0.5);
        }
    }

    let mut correct = 0usize;
    for i in 0..count {
        let out = ann.run(&input_data[i]);
        let predicted = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let actual = target_data[i].iter().position(|&v| v == 1.0).unwrap();
        if predicted == actual {
            correct += 1;
        }
    }

    println!(
        "Accuracy: {}/{} ({:.1}%)",
        correct,
        count,
        correct as f64 / count as f64 * 100.0
    );
}
