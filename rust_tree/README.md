# Rust-tree

R-tree implementation in rust.

Run program other than `main.rs`:
```
cargo run --bin fib
```

## Profiling

Use Hyperfine, install: 
```
sudo apt install build-essential
cargo install hyperfine
hyperfine --version
```

Usage:
```
hyperfine --warmup 10 --min-runs 25 -i 'cargo run --bin transpose'
```
