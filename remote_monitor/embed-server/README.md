# Mini Server for Raspberry Pi

## Deploy Steps

Using cross: `cargo install cross`

```sh
cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/embed-server "$HOST:/home/herb-remote-pi/embed-server"
scp .env "$HOST:/home/herb-remote-pi/embed-server"
```
