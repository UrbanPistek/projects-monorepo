# Using CUDA and the GPU to Parallelize Convolutional Neural Network Layers

## Summary 

The Rustacuda library was utilized to write the host to GPU interface. Additionally, 
kernel to run on the GPU were written in CUDA. The code optimized implementation of 
the CNN convolution, relu and output layers by running them on the GPU. The convolution and
relu layers ran together in the same kernel, the output was synchronized and passed to
the output layer kernel. Results were packaged and returned in a similar fashion as the
cpu implemenation. 

## Technical Details 

Firstly, the GPU device along with the kernel needed to be initialized on the host in Rust. 
The code in `cuda.rs` defines a CudaContext struct and its implementation. The CudaContext struct holds 
various fields such as `conv_layer`, `output_layer`, `module`, `stream`, and `device`. It initializes 
the rustacuda library, sets up the GPU device, and loads the compiled kernel from kernel.ptx file. 
The init function is used to initialize the context and the 
compute function runs the input through convolutional and output layers using the initialized context.

Cuda context setup in `init()` to inital the device and load the kernel module.
```
let cuda_context = Self {
	conv_layer: DeviceBox::new(&cnn.conv_layer).unwrap(),
	output_layer: DeviceBox::new(&cnn.output_layer).unwrap(),
	module: module,
	stream: stream,
	_context: ctx,
	device: cuda_device,
};
```

All the GPU parallelization is performed in `compute()`, firstly the input is placed into a 
`DeviceBox<>` so it can be passed to the GPU and `DeviceBuffer` objects are created to 
store data from the kernel. Since the output layer is fully connected - it requires all
convolution and relu computations to complete before executing. As a results, the kernels
were seperated into the convolution and relu kernel and the output kernel. The code blocks 
on the first launch to ensure all computations are completed before running the output layer. 

The two seperate kernels defined:
```
extern "C" __global__ void conv_layer(double input[100][100], double filter[10][5][5], double conv_output[10][20][20])

extern "C" __global__ void out_layer(double input[4000], double weights[10][4000], double output[10])
```

The convolution layer kernel is launch using 10 blocks and 400 threads, these values were chosen
since they made for easy indexing with the dimensions of the filter input and the convolution output. 
Since relu forces results to be `>= 0` this operation can be done in-place once the element-wise matrix 
product sum is computed between the input 5x5 window and the filter. For the output layer for simplicity 
the computations are parallelized on only 1 block across 10 threads to match with the dimensions of the weights
and the output. Overall, the convolution layer could be further improved by using a thread count which 
is a multiple of 32. The output layer has more potential for improved parallelization, however all
current attempts resulted in incorrect output - thus to perserve correctness the current implementation is used.

## Testing for Correctness 

To validate correctness both the gpu and cpu implementations were executed on the `input/in.csv` input. The
outputs of both were compared using the `compare.py` script. These results were validated on the 
`ecetesla2` and `eceubuntu4` servers, the `compare.py` script returned no discrepencies. 

```
eceubuntu4:~/performance_programming_ece459/a3_gpu_programming> cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv
    Finished release [optimized] target(s) in 0.69s
     Running `target/release/lab3 cpu input/cnn.csv input/in.csv output/out.csv`
82252 microseconds of actual work done

eceubuntu4:~/performance_programming_ece459/a3_gpu_programming> cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv
    Finished release [optimized] target(s) in 0.12s
     Running `target/release/lab3 cuda input/cnn.csv input/in.csv output/out_cuda.csv`
978345 microseconds of actual work done

eceubuntu4:~/performance_programming_ece459/a3_gpu_programming> python3 compare.py 
Comparison finished
```

```
ecetesla2:~/performance_programming_ece459/a3_gpu_programming> cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv
    Finished release [optimized] target(s) in 0.33s
     Running `target/release/lab3 cpu input/cnn.csv input/in.csv output/out.csv`
38117 microseconds of actual work done

ecetesla2:~/performance_programming_ece459/a3_gpu_programming> cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv
    Finished release [optimized] target(s) in 0.04s
     Running `target/release/lab3 cuda input/cnn.csv input/in.csv output/out_cuda.csv`
122876 microseconds of actual work done

ecetesla2:~/performance_programming_ece459/a3_gpu_programming> python3 compare.py 
Comparison finished
```

## Testing for Performance

Both implementations were tested using hyperfine on the `ecetesla2` server, the results are below: 

CPU:
```
Benchmark #1: cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv
  Time (mean ± σ):     244.5 ms ±  32.8 ms    [User: 195.5 ms, System: 29.4 ms]
  Range (min … max):   211.8 ms … 350.9 ms    25 runs
```

GPU:
```
Benchmark #1: cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv
  Time (mean ± σ):     505.4 ms ±  36.6 ms    [User: 291.6 ms, System: 168.4 ms]
  Range (min … max):   447.7 ms … 600.3 ms    25 runs
```

We can see that the GPU implementation is not faster than the CPU, form further testing
it was found that the output layer kernel is the main bottleneck. Form the analysis we can
see that the output layer is significantly slower. Overall, the output was not parallelized
in the most optimial fashion. However, for the scope of this project the correctness 
of the results was perserved. This is the main area of improvenment in term of performance.

```
Conv Layer Kernel: 10us
Output Layer Kernel: 369us

Conv Layer Kernel: 11us
Output Layer Kernel: 368us

Conv Layer Kernel: 11us
Output Layer Kernel: 366us

Conv Layer Kernel: 10us
Output Layer Kernel: 369us
```

