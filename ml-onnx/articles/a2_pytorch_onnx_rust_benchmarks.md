# Benchmarking the Performance of Model Inference in PyTorch & ONNX in Python and Rust

## [1] Overview

## [6] Summary

## [7] References

1. 

## Notes

Note: `_full` postfix -> profiling included the image pre-processing time. 

Profiling Commands

```
scalene onnx_inference.py
```

```
cargo flamegraph --bin onnx-inference
```

```
export MEMORY_PROFILER_LOG=warn
LD_PRELOAD=/home/urban/urban/projects/bytehound-0.11.0/target/release/libbytehound.so ./target/release/onnx-inference
/home/urban/urban/projects/bytehound-0.11.0/target/release/bytehound server memory-profiling_*.dat
```

#### Hyperfine

Python PyTorch
```
❯ hyperfine --warmup 2 --min-runs 5 'python pytorch_inference.py'                                                                                                                   100%
Benchmark 1: python pytorch_inference.py
  Time (mean ± σ):      3.214 s ±  0.178 s    [User: 3.863 s, System: 0.425 s]
  Range (min … max):    3.005 s …  3.442 s    5 runs
```

Python ONNX
```
❯ hyperfine --warmup 2 --min-runs 5 'python onnx_inference.py'                                                                                                                      100%
Benchmark 1: python onnx_inference.py
  Time (mean ± σ):     262.3 ms ±   9.2 ms    [User: 1433.6 ms, System: 88.7 ms]
  Range (min … max):   247.7 ms … 276.2 ms    11 runs
```

Rust - With Graph Level 3, THreads = 4
```
❯ hyperfine --warmup 2 --min-runs 5 'cargo run --release --bin onnx-inference'                                                                                                      100%
Benchmark 1: cargo run --release --bin onnx-inference
  Time (mean ± σ):     260.4 ms ±   5.4 ms    [User: 217.8 ms, System: 127.5 ms]
  Range (min … max):   251.9 ms … 269.3 ms    11 runs
```

Rust - With Graph Disabled, Threads = 4
```
❯ hyperfine --warmup 2 --min-runs 5 'cargo run --release --bin onnx-inference'                                                                                                      100%
Benchmark 1: cargo run --release --bin onnx-inference
  Time (mean ± σ):     234.4 ms ±  15.1 ms    [User: 232.0 ms, System: 109.0 ms]
  Range (min … max):   222.8 ms … 272.8 ms    10 runs
```

Rust - With Graph Disabled, Threads = 0
```
❯ hyperfine --warmup 2 --min-runs 5 'cargo run --release --bin onnx-inference'                                                                                                      100%
Benchmark 1: cargo run --release --bin onnx-inference
  Time (mean ± σ):     234.7 ms ±   6.9 ms    [User: 539.3 ms, System: 110.7 ms]
  Range (min … max):   224.0 ms … 243.0 ms    12 runs
```

Rust - With Graph Disabled, Threads = 8
```
❯ hyperfine --warmup 2 --min-runs 5 'cargo run --release --bin onnx-inference'                                                                                                      100%
Benchmark 1: cargo run --release --bin onnx-inference
  Time (mean ± σ):     228.8 ms ±   3.4 ms    [User: 366.0 ms, System: 106.8 ms]
  Range (min … max):   221.6 ms … 232.8 ms    13 runs
```
