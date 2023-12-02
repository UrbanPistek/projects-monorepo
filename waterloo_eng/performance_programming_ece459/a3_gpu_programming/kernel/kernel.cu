#include <stdio.h>

extern "C" __global__ void conv_layer(double input[100][100], double filter[10][5][5], double conv_output[10][20][20]) {

	int blockId = blockIdx.x; // size 10
	int threadId = threadIdx.x; // size 400

	// Preform the element-wise matrix multiplication
	double product = 0.0;
	for (int i = 0; i < 5; i++) {
        	for (int j = 0; j < 5; j++) {
            		product += input[(5*(threadId/20)) + i][(5*(threadId%20)) + j] * filter[blockId][i][j];
        	}
    	}

	// perform relu
	if (product < 0){
		product = 0;
	}

	// Store results in conv output layer
	conv_output[blockId][threadId/20][(threadId%20)] = product;
}

extern "C" __global__ void out_layer(double input[4000], double weights[10][4000], double output[10]) {

	int threadId = threadIdx.x; // size 10

	// Multiply conv output by each set of weights
	// Each thread calculates the dot product for that neuron
	for (int i = 0; i < 4000; i++){
		output[threadId] += input[i] * weights[threadId][i];
	}
}
