# Multi-threading with a Concurrent HashMap Implementation

## Summary

In a effort to increase performance, mutli-threading was implemented to parallelize the current bottleneck. 
Firstly, logic was added in `main.rs` to allow the program to run either using mutli-threading or a single thread
based on the inputs from the `--num-threads` argument. The bulk of the changes exist in `parsers.rs`, which primiraly
involved the implementation of functions `parse_raw_threaded_concurrent` - which minics the behaviour of `parse_raw` and
`dictionary_builder` along with `concurrent_dictionary_builder_line` - which minics the behaviour of `process_dictionary_builder_line`. 
These new functions implement the necessary changes to parallelze the operations in `dictionary_builder` using multi-threading.
This specfic implementation utilizes a concurrent HashMap using the DashMap library which is accessed by each thread to store the 2 
and 3 gram dictionaries. This implemeation runs by default and when the `--num-threads` arguement is specified. 

## Implementation Details

Some specfics are the same as the separate HashMap implemetation, notably the process of reading in the file and allocating chunks to each thread. 
The function `parse_raw_threaded_concurrent` was created to facilitate the concurrent map implemetation, its similar to `parse_raw_threaded_distributed`, 
however the key difference is that `parse_raw_threaded_concurrent` implements the concurrent HashMap - `DashMap<>` which does not require the use of 
mutexes. It only requires a `Atomic<>` type so that it can be referenced in each thread. Likewise, `concurrent_dictionary_builder_line` has similar 
functionality to `process_dictionary_builder_line` - however it requires that the inputs are of the `DashMap<>` type rather than a regular `HashMap<>`. 

```
fn concurrent_dictionary_builder_line(line: String, lookahead_line: Option<String>, regexp:&Regex, regexps:&Vec<Regex>, dbl: &mut Arc<DashMap<String, i32>>, trpl: &mut Arc<DashMap<String, i32>>, all_token_list: &mut Arc<Mutex<Vec<String>>>, prev1: Option<String>, prev2: Option<String>) -> (Option<String>, Option<String>)
```

Similar to `parse_raw_threaded_distributed` the `all_token_list` vector, is created as a `Arc<Mutex<Vec<String>>>` so that the Atomic reference 
can be passed into `concurrent_dictionary_builder_line`. As explained for the implementation of the separate HashMaps solution, this was done to 
improve performance. Similarily, a `Vec<thread::JoinHandle<()>>` is create to store the `JoinHandle` of each thread so they can be joined together 
once finished with execution. In this case, the `DashMap<>` needs to be created as a `Arc<>` type so that it can be referenced in each thread; the library
handles all the concurrency details. 

```
let dbl: Arc<DashMap<String, i32>> = Arc::new(DashMap::new()); 
let trpl: Arc<DashMap<String, i32>> = Arc::new(DashMap::new());
```

After all threads have been joined the only other key difference is that the two Atomic DashMaps need to be unwrapped so that they are
returned without the Atomic type. 

```
let final_dbl = Arc::try_unwrap(dbl).unwrap();
let final_trpl = Arc::try_unwrap(trpl).unwrap();
```

## Testing for Correctness

To test for the correctness, the single threaded implementation was executed, followed by this new implementation. The output of this implementation was 
comapred to the baseline single threaded one. As we can see from comparing the outputs, the size of all the dictionaries and token list is the same, and the
same set of dynamic tokens are gerenated for each method. 

### Baseline

Command: 

```
❯ cargo run --release -- --raw-hpc data/HPC_2k.log --to-parse "inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>" --before "running running" --after "configured out" --num-threads 0             
    Finished release [optimized] target(s) in 0.02s
     Running `target/release/logram --raw-hpc data/HPC_2k.log --to-parse 'inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>' --before 'running running' --after 'configured out' --num-threads 0`
```

Output: 

```
Num Threads: 0, Single Map: false

 Running on a single thread...

double dictionary list len 939, triple 1399, all tokens 484
["running", "running", "0xfffffffe", "<ok>", "node-1", "0xfffffffe", "<ok>", "node-2", "0xfffffffe", "<ok>", "node-30", "0xfffffffe", "<ok>", "configured", "out"]
2-gram node-30^0xfffffffe, count 1
2-gram 0xfffffffe^<ok>, count 4
2-gram <ok>^node-2, count 2
2-gram <ok>^configured, count 6
2-gram node-2^0xfffffffe, count 1
2-gram <ok>^node-30, count 1
2-gram node-1^0xfffffffe, count 1
2-gram <ok>^node-1, count 1
dynamic tokens: ["node-1", "node-2", "node-30"]
```

### Multi-threading with Concurrent Hashmaps Implementation

Command: 

```
❯ cargo run --release -- --raw-hpc data/HPC_2k.log --to-parse "inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>" --before "running running" --after "configured out" --num-threads 4 --single-map
    Finished release [optimized] target(s) in 0.02s
     Running `target/release/logram --raw-hpc data/HPC_2k.log --to-parse 'inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>' --before 'running running' --after 'configured out' --num-threads 4`
```

Output: 

```
Num Threads: 4, Single Map: false

 Running with multiple threads and a concurrent map...

double dictionary list len 939, triple 1399, all tokens 484
["running", "running", "0xfffffffe", "<ok>", "node-1", "0xfffffffe", "<ok>", "node-2", "0xfffffffe", "<ok>", "node-30", "0xfffffffe", "<ok>", "configured", "out"]
2-gram "0xfffffffe^<ok>", count 4
2-gram "<ok>^configured", count 6
2-gram "<ok>^node-1", count 1
2-gram "<ok>^node-30", count 1
2-gram "node-2^0xfffffffe", count 1
2-gram "node-1^0xfffffffe", count 1
2-gram "<ok>^node-2", count 2
2-gram "node-30^0xfffffffe", count 1
dynamic tokens: ["node-1", "node-2", "node-30"]
```

## Testing for Performance

Hyperfine was utilized to compared performance, the benchmarking was performed on mutliple different files. The results of particular set of benchmarks are
shown below.

Overall, benchmarking was performed on the same machine, in the same peroid to ensure minimal variance. Overall we can see that this implementation was
at most 2.7x faster than the single threaded execution. From testing, on the particular machine used, it was found that about 12-16 threads yielded maximun performance.
Any more threads resulted in a lower performance. 

### Baseline

Command: 

```
hyperfine --warmup 10 --min-runs 25 'cargo run --release -- --raw-linux data/Linux_2k_ext.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --num-threads 0'
```

Benchmark Results:

```
  Time (mean ± σ):     662.2 ms ±   6.6 ms    [User: 644.4 ms, System: 17.5 ms]
  Range (min … max):   652.6 ms … 676.5 ms    25 runs
```

### Multi-threading with Concurrent Hashmaps Implementation

Command: 

```
hyperfine --warmup 10 --min-runs 25 'cargo run --release -- --raw-linux data/Linux_2k_ext.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --num-threads 12'
```

Benchmark Results: 

```
  => threads = 2: 
  Time (mean ± σ):     462.0 ms ±  43.2 ms    [User: 801.3 ms, System: 23.1 ms]
  Range (min … max):   426.3 ms … 558.2 ms    25 runs
  
  => threads = 4:
  Time (mean ± σ):     323.5 ms ±  31.9 ms    [User: 937.0 ms, System: 23.6 ms]
  Range (min … max):   278.0 ms … 362.0 ms    25 runs

  => threads = 6:
  Time (mean ± σ):     281.2 ms ±  32.4 ms    [User: 1036.2 ms, System: 32.7 ms]
  Range (min … max):   229.6 ms … 342.4 ms    25 runs

  => threads = 8:
  Time (mean ± σ):     279.7 ms ±  29.8 ms    [User: 1199.8 ms, System: 39.1 ms]
  Range (min … max):   225.4 ms … 314.6 ms    25 runs

  => threads = 10:
  Time (mean ± σ):     260.6 ms ±  16.1 ms    [User: 1227.0 ms, System: 49.0 ms]
  Range (min … max):   211.0 ms … 282.9 ms    25 runs

  => threads = 12:
  Time (mean ± σ):     244.9 ms ±  21.2 ms    [User: 1224.7 ms, System: 50.2 ms]
  Range (min … max):   198.3 ms … 272.9 ms    25 runs
```

