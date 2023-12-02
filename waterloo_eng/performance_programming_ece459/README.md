# Performance Programming 
ECE 459 programming for performance. 

## Access UW Ubuntu Servers

First ssh into:
```
ssh upistek@eceterm.uwaterloo.ca
```

Then into either:
```
ssh upistek@ecetesla.uwaterloo.ca
ssh upistek@eceubuntu.uwaterloo.ca
```

## Using Hyperfine 

Can be used to benchmark rust programs: 

Install:

```
sudo apt install build-essential
cargo install hyperfine
hyperfine --version
```

Run: 

```
hyperfine '<commands>'
```

Ex: 

```
hyperfine 'cargo run --release'
```
