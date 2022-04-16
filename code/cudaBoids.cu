#include <algorithm>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h> 

#include "cudaBoids.h"

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {
    int imageWidth;
    int imageHeight;
    int boidCount;
    boid_t *boidData;
};

__constant__ GlobalConstants cuConstParams;


__global__ void kernelRender() {

}

////////////////////////////////////////////////////////////////////////////////////////


CudaBoids::CudaBoids() {
    // Initialize private variables
    image = NULL;
    boidCount = 0;
    cudaDeviceBoidData = NULL;
}

CudaBoids::~CudaBoids() {
    // Free allocated memory
    if (image) {
        delete image;
    }
    
    if (cudaDeviceBoidData) {
        cudaFree(cudaDeviceBoidData);
    }
}

void loadInput(const char *inputFile, Image *&image) {
    // Load the input into CPU memory
    FILE *input = fopen(inputFile, "r");
    if (!input) {
        printf("Unable to open file: %s.\n", inputFile);
        return;
    }

    int dim_x;
    int dim_y;
    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    image = new Image(dim_x, dim_y);
   
    int num_of_boids;
    fscanf(input, "%d\n", &num_of_boids);

    // Allocate mem for the boids
    image->data = (group_t*)malloc(sizeof(group_t));
    
    group_t *boid_group = image->data;
    boid_group->count = num_of_boids;
    
    boid_t *boids = (boid_t *)calloc(num_of_boids, sizeof(boid_t));
    boid_group->boids = boids;
    
    /* Read the grid dimension and boid information from file */
    
    // Load the coords (x1,y1) for each boid
    int x1, y1;
    for (int i = 0; i < num_of_boids; i++) {
        fscanf(input, "%d %d\n", &x1, &y1);
        boids[i].position.x = x1;
        boids[i].position.y = y1;
    }
}

void CudaBoids::setup(const char *inputName) {
    // First, load the input to CPU memory
    Image *image;
    loadInput(inputName, image);

    // Next, setup the cuda device
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaBoids\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // Copy data from CPU to GPU device memory
    boidCount = image->data->count;

    cudaMalloc(&cudaDeviceBoidData, sizeof(boid_t) * boidCount);

    cudaMemcpy(cudaDeviceBoidData, image->data->boids, sizeof(boid_t) * boidCount, cudaMemcpyHostToDevice);
   
    // Initialize parameters in constant memory
    GlobalConstants params;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.boidCount = boidCount;
    params.boidData = cudaDeviceBoidData;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

void CudaBoids::updateScene() {
    /*
    // 512 threads per block, and each thread is one pixel
    // so split image into blocks of 32x32 pixels
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((image->width + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (image->height + threadsPerBlock.y - 1)/threadsPerBlock.y);

    // Launch the main kernel
    kernelRender<<<numBlocks, threadsPerBlock>>>();
    */
}

Image *CudaBoids::output() {
    // Need to copy memory from GPU to CPU before returning Image ptr
    printf("Copying image data from device.\n");

    cudaMemcpy(image->data->boids, cudaDeviceBoidData, sizeof(boid_t) * boidCount, cudaMemcpyDeviceToHost);

    return image;
}
