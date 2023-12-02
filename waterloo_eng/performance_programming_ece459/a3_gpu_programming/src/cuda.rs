use crate::cnn::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::vec::Vec;

// Constants for thread and block sizes for each layer
// Convolution layer sizes
const CONV_NUM_BLOCKS:u32 = 10;
const CONV_NUM_THREADS:u32 = 400;

// Output layer sizes
const OUT_NUM_BLOCKS:u32 = 1;
const OUT_NUM_THREADS:u32 = 10;

// Constant for Kernel shared memory 
const KERNEL_SHARED_MEM_SIZE:u32 = 0;

// Use for debugging 
const DEBUG_LOGS:bool = false;

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,

    //Additional Fields
    #[allow(dead_code)]
    device: Device<>,
}

impl CudaContext {

    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {

        // Initialize rust cuda
        rustacuda::init(CudaFlags::empty()).unwrap();

        // Get GPU device
        let cuda_device: Device = Device::get_device(0).unwrap();
        if DEBUG_LOGS {println!("Using device: {}", cuda_device.name().unwrap());}

        // Configure GPU device
        let ctx: Context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, cuda_device).unwrap();

        // Get path to compiled kernel
        let ptx: CString = CString::new(include_str!("../kernel/kernel.ptx")).unwrap();

        // Load kernel
        let module: Module = Module::load_from_string(&ptx).unwrap();

        // Create stream
        let stream: Stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();


        // Update cuda context
        let cuda_context = Self {
            conv_layer: DeviceBox::new(&cnn.conv_layer).unwrap(),
            output_layer: DeviceBox::new(&cnn.output_layer).unwrap(),
            module: module,
            stream: stream,
            _context: ctx,
            device: cuda_device,
        };

        Ok(cuda_context)
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {

        // Setup input data to be passed into the kernel
        let mut input = DeviceBox::new(input).unwrap();

        // Store output of the convolution layer
        let conv_output_array = [[[0.0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE];
        let mut conv_output_buffer = DeviceBuffer::from_slice(&conv_output_array).unwrap();

        // Measure duration of kernel instances
        #[allow(unused_variables)]
        let mut ts = std::time::Instant::now();

        #[allow(unused_assignments)]
        let mut te = ts.elapsed().as_micros();

        // Load data into the kernel
        let stream = &self.stream;
        let cuda_module = &self.module;
        unsafe {
            let result = launch!(cuda_module.conv_layer<<<CONV_NUM_BLOCKS, CONV_NUM_THREADS, KERNEL_SHARED_MEM_SIZE, stream>>>(
                input.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_output_buffer.as_device_ptr()
            ));
            result?;
        }

        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        stream.synchronize().unwrap();

        // Log conv layer time
        te = ts.elapsed().as_micros();
        if DEBUG_LOGS {println!("Conv Layer Kernel: {}us", te);}

        // Create buffer to store resutls of the output layer
        let output_array = [0.0f64; OUT_LAYER_SIZE];
        let mut output_buffer = DeviceBuffer::from_slice(&output_array).unwrap();

        // Reset timer for next kernel
        ts = std::time::Instant::now();

        // Load output layer computations into kernel
        unsafe {
            let result = launch!(cuda_module.out_layer<<<OUT_NUM_BLOCKS, OUT_NUM_THREADS, KERNEL_SHARED_MEM_SIZE, stream>>>(
                conv_output_buffer.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_buffer.as_device_ptr()
            ));
            result?;
        }

        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        stream.synchronize().unwrap();

        // Log output layer time
        te = ts.elapsed().as_micros();
        if DEBUG_LOGS {println!("Output Layer Kernel: {}us", te);}

        // Copy final output
        let mut output_vec = [0.0f64; OUT_LAYER_SIZE];
        output_buffer.copy_to(&mut output_vec).unwrap();

        Ok(OutputVec(output_vec))

    }

    // Helper functions
    #[allow(dead_code)]
    fn flatten_conv_output_matrix(&mut self, vec2d: &[[f64;INPUT_DIM];INPUT_DIM]) -> Vec<f64> {
        let mut result:Vec<f64> = Vec::new();
        for row in vec2d {
            result.extend(row.iter());
        }
        result
    }
}
