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
