# Multi-threading with Separate Hashmaps Implementation

## Summary

In a effort to increase performance, mutli-threading was implemented to parallelize the current bottleneck. 
Firstly, logic was added in `main.rs` to allow the program to run either using mutli-threading or a single thread
based on the inputs from the `--num-threads` argument. The bulk of the changes exist in `parsers.rs`, which primiraly
involved the implementation of functions `parse_raw_threaded_distributed` - which minics the behaviour of `parse_raw` and
`dictionary_builder` and `distributed_dictionary_builder_line` - which minics the behaviour of `process_dictionary_builder_line`. 
These new functions implement the necessary changes to parallelze the operations in `dictionary_builder` using multi-threading. This
specific implementation allocated a seperate hash map to each thread and combined the results when all threads finished executing.
This implemetation runs when the `--single-maps` flag is specified along with `--num-threads` being set to greater than 1. 

## Implementation Details

Firstly, the bottleneck occurs at `dictionary_builder`; this function reads in the log file and then iterates over each line in the file. 
For each line the current and next line are passed into `process_dictionary_builder_line` which is responsible to construct the n-grams 
dictionaries. Since requirements stated that the main program is able to execute in any of the 3 modes, a separate function - `parse_raw_threaded_distributed` - was created
to facilitate this process, rather than making changes to `dictionary_builder`. Firstly, the entire file is read into memory and the contents 
are mapped to a `Vec<String>` where each element is a line from the file stored as a `String` type. This allows for the file to be separated 
into multiple chunks each of which can be allocated to a separate threads to operate on. The logic to separate the file into chunks is summarized below. 

```
let n = usize::try_from(*num_threads).unwrap();
let chunk_remainder = total_lines % n;
let portions = (total_lines - chunk_remainder) / n;

for i in 0..(*num_threads as usize) {

	// Get portion of file for this thread
	let file_chunk: Vec<String>;
	if i == (n-1) {
		file_chunk = parsed_lines[(i*portions)..(total_lines)].to_vec();
	} else {
		file_chunk = parsed_lines[(i*portions)..((i+1)*portions+1)].to_vec();
	}

	...
```

For each thread spawned based on the value of the `--num-threads` arguement the file chunk is moved into that thread. Additionally, before the spawning of each 
thread the HashMaps for dbl and trpl are created, these are created so that each the seperate hashmap created in each thread can merge together using the `extend()`
trait. These are create before the spawning of each thread so that they can outlive the lifetime of the threads. Further, the `all_token_list` vector, is created as a 
`Arc<Mutex<Vec<String>>>` type; this is done so that the Atomic reference can be passed into `distributed_dictionary_builder_line` which functions the same as
`process_dictionary_builder_line` except that it requires the Atomic Mutex in its arguements for `all_token_list`. This change is done so that the mutex 
can be locked and unlock with a limited scope only for specfic block of code where the `all_token_list` is accessed. This change was done to increase performance as
the `all_token_list` vector does not need to be copied into each thread and minimized scope of the mutex lock minimizes deadlock. Lastly, a `Vec<thread::JoinHandle<()>>` is
create to store the `JoinHandle` of each thread so they can be joined together once finished with execution. 

Lastly, within each thread a `dbl` and `trpl` HashMap are created to store the 2 and 3-grams, then the thread iterates over each line in the file chunk allocated to it. 
For each line it checks the value of the next line and based on the results it run `distributed_dictionary_builder_line` to create the 2/3-gram dictionaries. Once all the lines
in the file chunk have been processed, the results are added to the top level HashMaps as follows: 

```
// merge hashmaps
let mut dbl_lock = dbl_clone.lock().unwrap();
let mut trpl_lock = trpl_clone.lock().unwrap();

merge_hashmaps(&mut dbl_lock, &mut thread_dbl);
merge_hashmaps(&mut trpl_lock, &mut thread_trpl);
```

The following function was implemented to perform the merging of the two hashmaps. 

```
// Helper function to merge 2 hashmaps together
fn merge_hashmaps(base_map: &mut HashMap<String, i32>, map: &HashMap<String, i32>) {
    for (key, value) in map {
        match base_map.get_mut(key) {
            Some(x) => *x += value,
            None => {
                base_map.insert(key.to_string(), *value);
            }
        }
    }
}
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

### Multi-threading with Separate Hashmaps Implementation

Command: 

```
❯ cargo run --release -- --raw-hpc data/HPC_2k.log --to-parse "inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>" --before "running running" --after "configured out" --num-threads 4 --single-map
    Finished release [optimized] target(s) in 0.02s
     Running `target/release/logram --raw-hpc data/HPC_2k.log --to-parse 'inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>' --before 'running running' --after 'configured out' --num-threads 4 --single-map`
```

Output: 

```
Num Threads: 4, Single Map: true

 Running with multiple threads with seperate maps...

double dictionary list len 939, triple 1399, all tokens 484
["running", "running", "0xfffffffe", "<ok>", "node-1", "0xfffffffe", "<ok>", "node-2", "0xfffffffe", "<ok>", "node-30", "0xfffffffe", "<ok>", "configured", "out"]
2-gram 0xfffffffe^<ok>, count 4
2-gram <ok>^node-2, count 2
2-gram <ok>^node-1, count 1
2-gram node-1^0xfffffffe, count 1
2-gram node-2^0xfffffffe, count 1
2-gram <ok>^node-30, count 1
2-gram <ok>^configured, count 6
2-gram node-30^0xfffffffe, count 1
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

### Multi-threading with Separate Hashmaps Implementation

Command: 

```
hyperfine --warmup 10 --min-runs 25 'cargo run --release -- --raw-linux data/Linux_2k_ext.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --num-threads 12 --single-map'
```

Benchmark Results: 

```
  => threads = 4: 
  Time (mean ± σ):     429.6 ms ±  50.3 ms    [User: 735.5 ms, System: 23.3 ms]
  Range (min … max):   378.6 ms … 532.5 ms    25 runs
  
  => threads = 4:
  Time (mean ± σ):     309.9 ms ±  41.2 ms    [User: 863.6 ms, System: 28.6 ms]
  Range (min … max):   236.8 ms … 411.3 ms    25 runs

  => threads = 6:
  Time (mean ± σ):     256.7 ms ±  25.2 ms    [User: 895.3 ms, System: 32.4 ms]
  Range (min … max):   208.9 ms … 309.5 ms    25 runs

  => threads = 8:
  Time (mean ± σ):     272.9 ms ±  27.8 ms    [User: 1100.9 ms, System: 40.9 ms]
  Range (min … max):   216.7 ms … 303.3 ms    25 runs

  => threads = 10:
  Time (mean ± σ):     268.2 ms ±  18.3 ms    [User: 1347.1 ms, System: 47.2 ms]
  Range (min … max):   221.7 ms … 287.8 ms    25 runs

  => threads = 12:
  Time (mean ± σ):     264.5 ms ±  18.9 ms    [User: 1413.8 ms, System: 55.1 ms]
  Range (min … max):   217.6 ms … 284.2 ms    25 runs
```

