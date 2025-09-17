# ML ONNX

Dataset: [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

# Regnet Models

[Regnet](https://docs.pytorch.org/vision/main/models/regnet.html)

# Rust Notes

Install: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Update: `rustup update`

Initialize Directory: `cargo init postgresql-geospatial`

Add dependency: `cargo add sqlx@=0.8.6`

Commands:
```
cargo run
cargo build
cargo clean
cargo build --release
```

To build / run optimized release version add `--release` flag to those commands.

**Run sample:**

```
cargo run --bin rs_ort_sample
```

**Run Main:**

```
cargo run --bin onnx-inference
```

## Run flamegraph

```
sudo sysctl kernel.perf_event_paranoid=-1
cargo flamegraph -c "record -F 1000 -g --call-graph dwarf" --bin onnx-inference
```


## Run with hyperfine

```
hyperfine --warmup 2 --min-runs 5 'cargo run --bin onnx-inference --release'
hyperfine --warmup 2 --min-runs 5 'python pytorch_inference.py'
hyperfine --warmup 2 --min-runs 5 'python onnx_inference.py'
```

## Memory / CPU Profiling

### Python

[Scalene](https://github.com/plasma-umass/scalene)

```
scalene pytorch_inference.py
scalene onnx_inference.py
```

### Rust

CPU: [samply](https://github.com/mstange/samply)

```
cargo run --bin onnx-inference --release && samply record ./target/release/onnx-inference 
```

Memory: [bytehound](https://github.com/koute/bytehound)

Install: 

1. Install package from [here](https://github.com/koute/bytehound/releases), unzip locally, place in a location of your choice
2. Ensure gcc, rust >= 1.62 & [yarn](https://yarnpkg.com/) is installed 
3. Build package from the directory `cargo build --release -p bytehound-preload`

Run:

```
export MEMORY_PROFILER_LOG=warn
LD_PRELOAD=<path_to_bytehound_build>/libbytehound.so <path_to_target_binary>/onnx-inference
<path_to_bytehound_build>/bytehound server memory-profiling_*.dat
```

Sample: 

```
export MEMORY_PROFILER_LOG=warn
LD_PRELOAD=/home/urban/urban/projects/bytehound-0.11.0/target/release/libbytehound.so ./target/release/onnx-inference
/home/urban/urban/projects/bytehound-0.11.0/target/release/bytehound server memory-profiling_*.dat
```
