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

    int gridWidth;
    int gridHeight;
    int *gridIndex;

    int boidCount;
    int *hash;
    int *index;
    int *name;

    float *inData;
    float *outData;
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
__global__ void initIndex() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cuDataParams.boidCount)
        return;

    cuDataParams.index[i] = i;
    cuDataParams.name[i] = i;
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
    
    float2 *inData = (float2 *)&cuDataParams.inData;
    float2 pos = inData[2*i];
    float2 vel = inData[2*i+1];
   
    float2 accel = make_float2(0.f, 0.f);
    
    // Extra force (indep of neighbors) to stay on screen
    float2 centeringForce = make_float2(0.f, 0.f);

    float2 centerOffset = -pos;
    float t = sqrMagnitude(centerOffset) / cuDataParams.imageWidth;

    // t = 0 when at center, 1 if at edge, so start applying when 90% to edge
    if (t > 0.9f) {
        centeringForce = centerOffset * t * t;
    }

    float c = cuFlockParams.centeringWeight;
    if (sqrMagnitude(centeringForce) > c * c) {
        centeringForce = normalize(centeringForce);
        centeringForce *= c;
    }
 
    float2 cohesionForce = make_float2(0.f, 0.f);
    float2 alignmentForce = make_float2(0.f, 0.f);
    float2 separationForce = make_float2(0.f, 0.f);

    float2 averagePosition = make_float2(0.f, 0.f);
    float2 averageVelocity = make_float2(0.f, 0.f);

    int neighborCount = 0;
    int separateCount = 0;

    /* Calculate each individual force */
    for (int j = 0; j < boidCount; j++) {
        if (i != j) {
            float2 otherPos = inData[2*j];

            float dist = sqrDist(pos, otherPos);
            if (dist < cuFlockParams.squareNeighborRadius) {
                neighborCount++;

                // (Coherence) accumulate positions
                averagePosition += otherPos;
                
                // (Alignment) accumulate velocities
                float2 otherVel = inData[2*j+1];
                averageVelocity += otherVel;
                
                if (dist < cuFlockParams.squareAvoidanceRadius) {
                    separateCount++;
                    
                    // (Separation) accumulate weighted position offset
                    float2 offsetForce = pos - otherPos;
                    // Normalize and scale by distance (closest have most effect)
                    //offsetForce = normalize(offsetForce);
                    //offsetForce /= dist;

                    separationForce += offsetForce;
                }
            }
        }
    }
    
    // If there are no neighbors, maintain velocity & apply centering
    if (neighborCount == 0) {
        accel = centeringForce;
        // accel = vel + centeringForce;
        return accel;
    }
    
    // Average the forces
    if (neighborCount > 0) {
        averagePosition /= neighborCount;
        averageVelocity /= neighborCount;
    }
    if (separateCount > 0) {
        separationForce /= separateCount;
    }

    // Cohesion force is average position offset from current position
    cohesionForce = averagePosition - pos;
    // Alignment force is average velocity offset from current velocity
    alignmentForce = averageVelocity - vel;

    /* Weigh and combine the forces */
    float k = cuFlockParams.cohesionWeight;
    float m = cuFlockParams.alignmentWeight;
    float s = cuFlockParams.separationWeight;

    cohesionForce *= k;
    alignmentForce *= m;
    separationForce *= s;

    if (sqrMagnitude(cohesionForce) > k * k) {
        cohesionForce = normalize(cohesionForce);
        cohesionForce *= k;
    }
    if (sqrMagnitude(alignmentForce) > m * m) {
        alignmentForce = normalize(alignmentForce);
        alignmentForce *= m;
    }
    if (sqrMagnitude(separationForce) > s * s) {
        separationForce = normalize(separationForce);
        separationForce *= s;
    }

    accel = cohesionForce + alignmentForce + separationForce + centeringForce;
   
    /* 
    if (i != 0)
        return accel;

    printf("Boid %d has accel %lf %lf: cohesion %lf %lf, align %lf %lf, separate %lf %lf, center %lf %lf.\n",
            i, accel.x, accel.y, cohesionForce.x, cohesionForce.y, alignmentForce.x, alignmentForce.y,
            separationForce.x, separationForce.y, centeringForce.x, centeringForce.y);
    printf("Boid %d had %d boids in neighbor radius2 of %lf and %d boids in avoid radius2 of %lf.\n",
            i, neighborCount, cuFlockParams.squareNeighborRadius,
            separateCount, cuFlockParams.squareAvoidanceRadius);
    */

    /* Return the resulting acceleration (for this frame) */
    return accel;
}

__global__ void moveBoids() {
    // WARNING: Method is currently very inefficient, O(n^2) neighbor check
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int boidCount = cuDataParams.boidCount;
    if (i >= boidCount)
        return;

    /* Calculate the combined steers from all forces */
    float2 accel = calculate_move(i);
    
    // scale by drive factor
    accel *= cuFlockParams.driveFactor;

    // cap at max speed
    if (sqrMagnitude(accel) > cuFlockParams.squareMaxSpeed) {
        accel = normalize(accel);
        accel *= cuFlockParams.maxSpeed;
    }

    /* Apply the steers to move position */
    float2 oldVelocity = ((float2 *)&cuDataParams.inData)[2*i+1];
    float2 newVelocity = oldVelocity + accel;

    float2 oldPosition = ((float2 *)&cuDataParams.inData)[2*i];
    float2 newPosition = oldPosition + newVelocity;

    /*
    if (i == 0) {
        printf("Boid %d old pos %lf %lf vel %lf %lf, update to pos %lf %lf vel %lf %lf.\n",
                i, oldPosition.x, oldPosition.y, oldVelocity.x, oldVelocity.y,
                newPosition.x, newPosition.y, newVelocity.x, newVelocity.y);
    }
    */
    ((float2 *)cuDataParams.outData)[2*i] = newPosition;
    ((float2 *)cuDataParams.outData)[2*i+1] = newVelocity;
}

__global__ void kernelPrint() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i >= cuDataParams.boidCount)
        return;

    /*
    boid_t boid = cuDataParams.inData[i];
    printf("boid %d indata is %lf %lf %lf %lf\n", i,
            boid.position.x, boid.position.y, boid.velocity.x, boid.velocity.y);
    boid = cuDataParams.outData[i];
    printf("boid %d outdata is %lf %lf %lf %lf\n", i,
            boid.position.x, boid.position.y, boid.velocity.x, boid.velocity.y);
    */

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

__global__ void updateGrid() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int boidCount = cuDataParams.boidCount;
    if (i >= boidCount)
        return;

    // TODO: Load this section of grid index into shared mem
    int *gridIndex = cuDataParams.gridIndex;

    if (i == 0) {
        // First grid square always starts with 0
        gridIndex[0] = 0;
        return;
    }
    
    // Update the grid index: check if this boid is start of a grid
    int myGridIndex = gridIndex[i];
    int prevGridIndex = gridIndex[i-1];

    if (prevGridIndex != myGridIndex) {
        // Indicate that this index is the start of new grid square
        gridIndex[myGridIndex] = i;
    }

    // Load this boid's data into input in grid sorted order
    int index = cuDataParams.index[i];
    float4 boidData = ((float4 *)cuDataParams.outData)[i];
    ((float4 *)cuDataParams.inData)[index] = boidData;
}

__global__ void bitonicSwap(int j, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = i ^ j;
    
    if (i <= l)
        return;

    // TODO: Load into shared memory then push to global together?
    int *hash = cuDataParams.hash;
    int *index = cuDataParams.index;
    int *name = cuDataParams.name;

    bool nonincr = (i & k) == 0;
    if ((nonincr && hash[l] < hash[i]) || (!nonincr && hash[l] > hash[i])) {
        // Swap the elements i,l for hash and index arays
        int temp = hash[i];
        hash[i] = hash[l];
        hash[l] = temp;

        temp = index[i];
        index[i] = index[l];
        index[l] = temp;

        temp = name[i];
        name[i] = name[l];
        name[l] = temp;
    }
}

__device__ __inline__ int mortonCode(unsigned int x, unsigned int y) {
    int code = 0;
    unsigned int mask = 0b1;
    int numShifts = (cuDataParams.gridWidth * cuDataParams.gridHeight / 2) * 8;
    
    for (int i = 0; i < numShifts; i++) {
        code |= ((x & mask) << 2*i) | ((y & mask) << (2*i + 1));
        mask = mask << 1;
    }

    return code;
}

__global__ void updateHash() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int boidCount = cuDataParams.boidCount;
    if (i >= boidCount)
        return;

    // TODO: Load this block's positions into shared memory?

    // Hash position of ith boid in output array
    float2 position = ((float2 *)cuDataParams.outData)[2*i];
    unsigned int gridX = (unsigned int)(position.x / cuDataParams.gridWidth);
    unsigned int gridY = (unsigned int)(position.y / cuDataParams.gridHeight);
    int hash = mortonCode(gridX, gridY);

    // Store the calculated hash
    cuDataParams.hash[i] = hash;
    
    // Update the index to match hash
    cuDataParams.index[i] = i;
}

///////////////////////////////////////////////////////////////////////////////////////
CudaBoids::CudaBoids() {
    image = NULL;
    gridWidth = 0;
    gridHeight = 0;
    deviceGridIndex = NULL;

    boidCount = 0;
    deviceHash = NULL;
    deviceIndex = NULL;
    deviceName = NULL;
    deviceInData = NULL;
    deviceOutData = NULL;
}

CudaBoids::~CudaBoids() {
    // Free allocated memory
    if (image) {
        delete image;
    }
    
    if (deviceInData) {
        cudaFree(deviceHash);
        cudaFree(deviceIndex);
        cudaFree(deviceName);
        cudaFree(deviceGridIndex);
        cudaFree(deviceInData);
        cudaFree(deviceOutData);
    }
}

void bitonicSort(int count) {
    for (int k = 2; k <= count; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSwap<<<1, count>>>(j, k);
            cudaDeviceSynchronize();
        }
    }
}

void CudaBoids::updateScene() {
    dim3 blockDim(256, 1);
    dim3 gridDim((boidCount + blockDim.x - 1) / blockDim.x);

    // TODO:
    // Get hashes for all boids,
    // sort by their hash value,
    // populate the gridIndexes,
    // copy and order the float data by the hash order
    // Get the hash for the new position of the boids
    updateHash<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    // Re-order boids based on their hash (bitonic sort)
    // TODO: Power of 2 arrays
    bitonicSort(256);
    cudaDeviceSynchronize();

    // Set grid start/end, load data into input in hash-sorted order
    updateGrid<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    // The main boids animation, calculate forces then update velocity/position
    moveBoids<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
   
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();
}

Image *CudaBoids::output() {
    // Need to copy memory from GPU to CPU before returning Image ptr
    // TODO: Need to re-order to match input index instead of just memcpy
    
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

    // Define the flock parameter constants
    // These two values are the same, convenient to have as int and float
    int viewRadius = 80;
    float neighborRadius = 80.f;

    GlobalFlockConstants flockParams;
    //flockParams.driveFactor = 10.f;
    flockParams.driveFactor = 2.f;
    //flockParams.maxSpeed = 5.f;
    flockParams.maxSpeed = 15.f;
    flockParams.squareMaxSpeed = flockParams.maxSpeed * flockParams.maxSpeed;
    //flockParams.squareNeighborRadius = 1.5f * 1.5f;
    flockParams.squareNeighborRadius = neighborRadius * neighborRadius;
    flockParams.squareAvoidanceRadius = flockParams.squareNeighborRadius * 0.75f * 0.75f;
    flockParams.cohesionWeight = 1.f;
    flockParams.alignmentWeight = 1.f;
    flockParams.separationWeight = 1.f;
    flockParams.centeringWeight = 0.05f;

    cudaMemcpyToSymbol(cuFlockParams, &flockParams, sizeof(GlobalFlockConstants));
 
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();

    // Copy/create necessary memory allocs on the GPU   
    gridWidth = (image->width + viewRadius - 1) / viewRadius;
    gridHeight = (image->height + viewRadius - 1) / viewRadius; 
    cudaMalloc(&deviceGridIndex, sizeof(int) * gridWidth * gridHeight);
    
    boidCount = image->data->count;
    cudaMalloc(&deviceHash, sizeof(int) * boidCount);
    cudaMalloc(&deviceIndex, sizeof(int) * boidCount);
    cudaMalloc(&deviceName, sizeof(int) * boidCount);
    cudaMalloc(&deviceInData, sizeof(float4) * boidCount);
    cudaMalloc(&deviceOutData, sizeof(float4) * boidCount);
    
    // TODO: Does this actually work?
    cudaMemcpy(deviceInData, image->data->boids, sizeof(boid_t) * boidCount, cudaMemcpyHostToDevice);

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // Initialize parameters in constant memory
    GlobalDataConstants dataParams;
    dataParams.imageWidth = image->width;
    dataParams.imageHeight = image->height;
    dataParams.gridWidth = gridWidth;
    dataParams.gridHeight = gridHeight;
    dataParams.gridIndex = deviceGridIndex;
    dataParams.boidCount = boidCount;
    dataParams.hash = deviceHash;
    dataParams.index = deviceIndex;
    dataParams.name = deviceName;
    dataParams.inData = deviceInData;
    dataParams.outData = deviceOutData;

    cudaMemcpyToSymbol(cuDataParams, &dataParams, sizeof(GlobalDataConstants));

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    initIndex<<<1, boidCount>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}
