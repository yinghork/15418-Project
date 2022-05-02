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

struct GlobalDataConstants {
    int imageWidth;
    int imageHeight;

    int boidCount;
    boid_t *inData;
    boid_t *outData;
};

struct GlobalFlockConstants {
    float driveFactor;
    float maxSpeed;
    float squareMaxSpeed;
    float squareNeighborRadius;
    float squareAvoidanceRadius;

    float cohesionWeight;
    float alignmentWeight;
    float separationWeight;
    float centeringWeight;
};

__constant__ GlobalDataConstants cuDataParams;
__constant__ GlobalFlockConstants cuFlockParams;

////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFrame() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cuDataParams.boidCount)
        return;

    cuDataParams.inData[i] = cuDataParams.outData[i];
}

// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h
inline __host__ __device__ float2 operator-(float2 &a) {
    return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(float2 &a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline __host__ __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ void operator*=(float2 &a, float b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__ void operator/=(float2 &a, float b) {
    a.x /= b;
    a.y /= b;
}


// My helper functions
__device__ __inline__ float2 toFloat2(pos_t pos) {
    return make_float2(pos.x, pos.y);
}

__device__ __inline__ float2 toFloat2(vel_t vel) {
    return make_float2(vel.x, vel.y);
}

__device__ __inline__ pos_t toPos(float2 f) {
    pos_t pos;
    pos.x = f.x;
    pos.y = f.y;
    return pos;
}

__device__ __inline__ vel_t toVel(float2 f) {
    vel_t vel;
    vel.x = f.x;
    vel.y = f.y;
    return vel;
}

// Float math helper functions
__device__ __inline__ float sqrMagnitude(float2 f) {
    return f.x * f.x + f.y * f.y;
}

__device__ __inline__ float sqrDist(float2 f1, float2 f2) {
    float distX = f1.x - f2.x;
    float distY = f1.y - f2.y;
    return distX * distX + distY * distY;
}

__device__ __inline__ float2 normalize(float2 f) {
    float magnitude = sqrt(f.x * f.x + f.y * f.y);
    return f / magnitude;
}

// Calculate all the steers to apply to the boid
__device__ __inline__ float2 calculate_move(int i) {
    int boidCount = cuDataParams.boidCount;
    
    boid_t *inData = cuDataParams.inData;
    float2 pos = toFloat2(inData[i].position);
    float2 vel = toFloat2(inData[i].velocity);
   
    float2 move = make_float2(0.f, 0.f);
    
    // Extra force (indep of neighbors) to stay on screen
    float2 ceMove = make_float2(0.f, 0.f);

    float2 centerOffset = -pos;
    float t = sqrMagnitude(centerOffset) / cuDataParams.imageWidth;

    // t = 0 when at center, 1 if at edge, so start applying when 90% to edge
    if (t > 0.81f) {
        ceMove *= t * t;
    }

    float c = cuFlockParams.centeringWeight;
    if (sqrMagnitude(ceMove) > c * c) {
        ceMove = normalize(ceMove);
        ceMove *= c;
    }
 
    float2 coMove = make_float2(0.f, 0.f);
    float2 alMove = make_float2(0.f, 0.f);
    float2 seMove = make_float2(0.f, 0.f);

    int neighborCount = 0;
    int separateCount = 0;

    /* Calculate each individual force */
    for (int j = 0; j < boidCount; j++) {
        if (i != j) {
            boid_t otherBoid = inData[j];
            float2 otherPos = toFloat2(otherBoid.position);
            
            float dist = sqrDist(pos, otherPos);
            if (dist < cuFlockParams.squareNeighborRadius) {
                neighborCount++;

                // (Coherence) accumulate positions
                coMove += otherPos;
                
                // (Alignment) accumulate velocities
                alMove += toFloat2(otherBoid.velocity);
                
                if (dist < cuFlockParams.squareAvoidanceRadius) {
                    separateCount++;

                    // (Separation) accumulate position offset
                    seMove += (pos - otherPos);
                }
            }
        }
    }
    
    // If there are no neighbors, maintain velocity & apply centering
    if (neighborCount == 0) {
        move = vel + ceMove;
        return move;
    }
    
    // Average the forces
    if (neighborCount > 0) {
        coMove /= neighborCount;
        alMove /= neighborCount;
    }
    if (separateCount > 0) {
        seMove /= separateCount;
    }

    // Convert cohesion to an offset from current position
    coMove -= pos;
    // Convert alignment to offset from current velocity
    alMove -= vel;

    /* Weigh and combine the forces */
    float k = cuFlockParams.cohesionWeight;
    float m = cuFlockParams.alignmentWeight;
    float s = cuFlockParams.separationWeight;

    coMove *= k;
    alMove *= m;
    seMove *= s;

    // Start combining forces
    if (sqrMagnitude(coMove) > k * k) {
        coMove = normalize(coMove);
        coMove *= k;
    }
    if (sqrMagnitude(alMove) > m * m) {
        alMove = normalize(alMove);
        alMove *= m;
    }
    if (sqrMagnitude(seMove) > s * s) {
        seMove = normalize(seMove);
        seMove *= s;
    }

    move += coMove + alMove + seMove + ceMove;
   
    /* 
    if (i != 0)
        return move;

    printf("Boid %d has move %lf %lf: cohere %lf %lf, align %lf %lf, avoid %lf %lf, center %lf %lf.\n",
            i, move.x, move.y, coMove.x, coMove.y, alMove.x, alMove.y, seMove.x, seMove.y, ceMove.x, ceMove.y);
    printf("Boid %d had %d boids in neighborhood of %lf and %d boids in avoid radius of %lf.\n",
            i, neighborCount, cuFlockParams.squareNeighborRadius,
            separateCount, cuFlockParams.squareAvoidanceRadius);
    */

    /* Return the resulting composition move */
    return move;
}

__global__ void moveBoids() {
    // WARNING: Method is currently very inefficient, O(n^2) neighbor check
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int boidCount = cuDataParams.boidCount;
    if (i >= boidCount)
        return;

    /* Calculate the combined steers from all forces */
    float2 move = calculate_move(i);
    
    // scale by drive factor
    move *= cuFlockParams.driveFactor;

    // cap at max speed
    if (sqrMagnitude(move) > cuFlockParams.squareMaxSpeed) {
        move = normalize(move);
        move *= cuFlockParams.maxSpeed;
    }

    /* Apply the steers to move position */
    float2 newVelocity = toFloat2(cuDataParams.inData[i].velocity);
    newVelocity += move;
    float2 oldPosition = toFloat2(cuDataParams.inData[i].position);
    float2 newPosition = oldPosition + newVelocity;

    cuDataParams.outData[i].velocity = toVel(newVelocity);
    cuDataParams.outData[i].position = toPos(newPosition);
}

__global__ void kernelPrint() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i >= cuDataParams.boidCount)
        return;

    boid_t boid = cuDataParams.inData[i];
    printf("boid %d indata is %lf %lf %lf %lf\n", i,
            boid.position.x, boid.position.y, boid.velocity.x, boid.velocity.y);
    boid = cuDataParams.outData[i];
    printf("boid %d outdata is %lf %lf %lf %lf\n", i,
            boid.position.x, boid.position.y, boid.velocity.x, boid.velocity.y);

    if (i != 0)
        return;

    printf("Flock Constants:\n");
    printf("driveFactor is %lf\n", cuFlockParams.driveFactor);
    printf("maxSpeed is %lf\n", cuFlockParams.maxSpeed);
    printf("cohesion weight is %lf\n", cuFlockParams.cohesionWeight);
    printf("alignment weight is %lf\n", cuFlockParams.alignmentWeight);
    printf("avoidance weight is %lf\n", cuFlockParams.separationWeight);

    printf("\nData Constants:\n");
    printf("boidCount is %d\n", cuDataParams.boidCount);
    printf("imageWidth is %d\n", cuDataParams.imageWidth);
    printf("imageHeight is %d\n", cuDataParams.imageHeight); 
}

__global__ void kernelPrintPrivate(boid_t *deviceData) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0)
        return;

    printf("Device data:\n");
    printf("first boid at data addr is %lf %lf %lf %lf\n",
            deviceData[0].position.x, deviceData[0].position.y, 
            deviceData[0].velocity.x, deviceData[0].velocity.y);
}

///////////////////////////////////////////////////////////////////////////////////////
CudaBoids::CudaBoids() {
    image = NULL;
    boidCount = 0;
    deviceInData = NULL;
    deviceOutData = NULL;
}

CudaBoids::~CudaBoids() {
    // Free allocated memory
    if (image) {
        delete image;
    }
    
    if (deviceInData) {
        cudaFree(deviceInData);
        cudaFree(deviceOutData);
    }
}

void CudaBoids::updateScene() {
    dim3 blockDim(256, 1);
    dim3 gridDim((boidCount + blockDim.x - 1) / blockDim.x);

    moveBoids<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
  
    // Now that finished with input data, overwrite with output data for next frame
    copyFrame<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
   
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();
}

Image *CudaBoids::output() {
    // Need to copy memory from GPU to CPU before returning Image ptr
    cudaMemcpy(image->data->boids, deviceOutData, sizeof(boid_t) * boidCount, cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    return image;
}

void CudaBoids::setup(const char *inputName, int num_of_threads) {
    // First, load the input to CPU memory
    FILE *input = fopen(inputName, "r");
    if (!input) {
        printf("Unable to open file: %s.\n", inputName);
        return;
    }

    image = (Image *)malloc(sizeof(Image));

    int dim_x;
    int dim_y;
    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    
    int num_of_boids;
    fscanf(input, "%d\n", &num_of_boids);

    image->width = 2 * dim_x;
    image->height = 2 * dim_y;

    image->data = (group_t*)malloc(sizeof(group_t));
    
    image->data->count = num_of_boids;
    image->data->boids = (boid_t *)calloc(num_of_boids, sizeof(boid_t));

    /* Read the grid dimension and boid information from file */
    
    // Load the coords (x1,y1) for each boid
    int x1, y1;
    for (int i = 0; i < num_of_boids; i++) {
        fscanf(input, "%d %d\n", &x1, &y1);
        image->data->boids[i].position.x = (float)x1;
        image->data->boids[i].position.y = (float)y1;
        image->data->boids[i].velocity.x = (rand() % 3) - 1.f;
        image->data->boids[i].velocity.y = (rand() % 3) - 1.f;
    }

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

    // Copy/create necessary data on the GPU
    boidCount = image->data->count;

    cudaMalloc(&deviceInData, sizeof(boid_t) * boidCount);
    cudaMalloc(&deviceOutData, sizeof(boid_t) * boidCount);
    
    cudaMemcpy(deviceInData, image->data->boids, sizeof(boid_t) * boidCount, cudaMemcpyHostToDevice);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // Initialize parameters in constant memory
    GlobalDataConstants dataParams;
    dataParams.imageWidth = image->width;
    dataParams.imageHeight = image->height;
    dataParams.boidCount = boidCount;
    dataParams.inData = deviceInData;
    dataParams.outData = deviceOutData;

    cudaMemcpyToSymbol(cuDataParams, &dataParams, sizeof(GlobalDataConstants));

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    GlobalFlockConstants flockParams;
    //flockParams.driveFactor = 10.f;
    flockParams.driveFactor = 10.f;
    //flockParams.maxSpeed = 5.f;
    flockParams.maxSpeed = 5.f;
    flockParams.squareMaxSpeed = flockParams.maxSpeed * flockParams.maxSpeed;
    //flockParams.squareNeighborRadius = 1.5f * 1.5f;
    flockParams.squareNeighborRadius = 75.f * 75.f;
    flockParams.squareAvoidanceRadius = flockParams.squareNeighborRadius * 0.5f * 0.5f;
    flockParams.cohesionWeight = 1.f;
    flockParams.alignmentWeight = 1.f;
    flockParams.separationWeight = 1.f;
    flockParams.centeringWeight = 0.1f;

    cudaMemcpyToSymbol(cuFlockParams, &flockParams, sizeof(GlobalFlockConstants));
 
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();
}
