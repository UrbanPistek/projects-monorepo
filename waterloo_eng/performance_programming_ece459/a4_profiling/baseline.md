# Baseline

Command: 

(local)
```
hyperfine --warmup 5 --min-runs 25 -i 'target/release/lab4'
```

Output (on defaults):

```
Benchmark 1: target/release/lab4
  Time (mean ± σ):      97.4 ms ±   7.6 ms    [User: 698.8 ms, System: 88.3 ms]
  Range (min … max):    82.0 ms … 110.7 ms    36 runs
```

Perf stat: 
```
 Performance counter stats for 'cargo run --release':

            781.43 msec task-clock                #    5.595 CPUs utilized          
             1,288      context-switches          #    1.648 K/sec                  
                37      cpu-migrations            #   47.349 /sec                   
             4,534      page-faults               #    5.802 K/sec                  
     2,200,620,473      cycles                    #    2.816 GHz                      (82.65%)
        87,253,242      stalled-cycles-frontend   #    3.96% frontend cycles idle     (80.79%)
       357,669,169      stalled-cycles-backend    #   16.25% backend cycles idle      (82.46%)
     2,645,706,878      instructions              #    1.20  insn per cycle         
                                                  #    0.14  stalled cycles per insn  (84.28%)
       652,017,999      branches                  #  834.394 M/sec                    (85.62%)
         5,320,469      branch-misses             #    0.82% of all branches          (86.53%)

       0.139669578 seconds time elapsed

       0.733643000 seconds user
       0.048372000 seconds sys
```
