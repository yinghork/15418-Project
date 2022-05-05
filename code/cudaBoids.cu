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
    float4 *inData;
    float4 *outData;
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
    // TODO: Consider moving into update method, for shared block mem?

    int boidCount = cuDataParams.boidCount;
    
    float4 *inData = cuDataParams.inData;
    float4 data = inData[i];
    float2 pos = make_float2(data.x, data.y);
    float2 vel = make_float2(data.z, data.w);
   
    float2 accel = make_float2(0.f, 0.f);
    
    // TODO: Do we keep the centering force? Leader following?
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
 
    // Calculate the basic 3 forces
    float2 cohesionForce = make_float2(0.f, 0.f);
    float2 alignmentForce = make_float2(0.f, 0.f);
    float2 separationForce = make_float2(0.f, 0.f);

    float2 averagePosition = make_float2(0.f, 0.f);
    float2 averageVelocity = make_float2(0.f, 0.f);

    int neighborCount = 0;
    int separateCount = 0;

    // TODO: Load a block of boids into shared mem

    /* Calculate each individual force */
    for (int j = 0; j < boidCount; j++) {
        if (i != j) {
            float4 otherBoid = inData[j];
            float2 otherPos = make_float2(otherBoid.x, otherBoid.y);
            
            float dist = sqrDist(pos, otherPos);
            if (dist < cuFlockParams.squareNeighborRadius) {
                neighborCount++;

                // (Coherence) accumulate positions
                averagePosition += otherPos;
                
                // (Alignment) accumulate velocities
                averageVelocity += make_float2(otherBoid.z, otherBoid.w);
                
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

    // TODO: Load locally useful constants
    // Load shared mem of these boids?

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
    // Format the global loads/stores as float4 for max efficiency
    float4 oldData = cuDataParams.inData[i];
    
    float2 oldVelocity = make_float2(oldData.z, oldData.w);
    float2 newVelocity = oldVelocity + accel;

    float2 oldPosition = make_float2(oldData.x, oldData.y);
    float2 newPosition = oldPosition + newVelocity;

    float4 newData = make_float4(newPosition.x, newPosition.y, newVelocity.x, newVelocity.y);
    cuDataParams.outData[i] = newData;

    /*
    if (i == 0) {
        printf("Boid %d old pos %lf %lf vel %lf %lf, update to pos %lf %lf vel %lf %lf.\n",
                i, oldPosition.x, oldPosition.y, oldVelocity.x, oldVelocity.y,
                newPosition.x, newPosition.y, newVelocity.x, newVelocity.y);
    }
    */
}

__global__ void copyFrame() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cuDataParams.boidCount)
        return;

    cuDataParams.inData[i] = cuDataParams.outData[i];
}

__global__ void kernelPrint() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i >= cuDataParams.boidCount)
        return;

    float4 boid = ((float4 *)cuDataParams.inData)[i];
    printf("boid %d indata is %lf %lf %lf %lf\n", i,
            boid.x, boid.y, boid.z, boid.w);
    boid = ((float4 *)cuDataParams.outData)[i];
    printf("boid %d outdata is %lf %lf %lf %lf\n", i,
            boid.x, boid.y, boid.z, boid.w);

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

///////////////////////////////////////////////////////////////////////////////////////
CudaBoids::CudaBoids() {
    image = NULL;
    hostData = NULL;
    boidCount = 0;
    deviceInData = NULL;
    deviceOutData = NULL;
}

CudaBoids::~CudaBoids() {
    // Free allocated memory
    if (image) {
        delete image;
        delete hostData;
    }
    
    if (deviceInData) {
        cudaFree(deviceInData);
        cudaFree(deviceOutData);
    }
}

void CudaBoids::updateScene() {
    dim3 blockDim(256, 1);
    dim3 gridDim((boidCount + blockDim.x - 1) / blockDim.x);

    // Copy data from previous frame output to the input
    // TODO: Should we just make this a memcpy if we aren't going to reorganize?
    copyFrame<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    // Update the boids based on new input data
    moveBoids<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
  
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();
}

void copyBoidToFloat(boid_t *src, float4 *dst, int count) {
    // Reformat boid's pos and vel as float4, then store
    boid_t *boid;
    float4 boidData;
    for (int i = 0; i < count; i++) {
        boid = &(src[i]);
        boidData = make_float4(boid->position.x, boid->position.y, boid->velocity.x, boid->velocity.y);
        dst[i] = boidData;
    }
}

void copyFloatToBoid(float4 *src, boid_t *dst, int count) {
    // Extract float2 from float4 data and format for struct
    boid_t *boid;
    float4 boidData;
    for (int i = 0; i < count; i++) {
        boidData = src[i];
        boid = &(dst[i]);
        boid->position.x = boidData.x;
        boid->position.y = boidData.y;
        boid->velocity.x = boidData.z;
        boid->velocity.y = boidData.w;
    }
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

    // Create the float4 formatted version of image data
    hostData = (float *)calloc(num_of_boids, sizeof(float4));
    copyBoidToFloat(image->data->boids, (float4 *)hostData, num_of_boids);

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

    cudaMalloc(&deviceInData, sizeof(float4) * boidCount);
    cudaMalloc(&deviceOutData, sizeof(float4) * boidCount);
    
    cudaMemcpy(deviceOutData, hostData, sizeof(float4) * boidCount, cudaMemcpyHostToDevice);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // Initialize parameters in constant memory
    GlobalDataConstants dataParams;
    dataParams.imageWidth = image->width;
    dataParams.imageHeight = image->height;
    dataParams.boidCount = boidCount;
    dataParams.inData = (float4 *)deviceInData;
    dataParams.outData = (float4 *)deviceOutData;

    cudaMemcpyToSymbol(cuDataParams, &dataParams, sizeof(GlobalDataConstants));

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    GlobalFlockConstants flockParams;
    //flockParams.driveFactor = 10.f;
    flockParams.driveFactor = 2.f;
    //flockParams.maxSpeed = 5.f;
    flockParams.maxSpeed = 15.f;
    flockParams.squareMaxSpeed = flockParams.maxSpeed * flockParams.maxSpeed;
    //flockParams.squareNeighborRadius = 1.5f * 1.5f;
    flockParams.squareNeighborRadius = 80.f * 80.f;
    flockParams.squareAvoidanceRadius = flockParams.squareNeighborRadius * 0.75f * 0.75f;
    flockParams.cohesionWeight = 1.f;
    flockParams.alignmentWeight = 1.f;
    flockParams.separationWeight = 1.f;
    flockParams.centeringWeight = 0.05f;

    cudaMemcpyToSymbol(cuFlockParams, &flockParams, sizeof(GlobalFlockConstants));
 
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize();
}

Image *CudaBoids::output() {
    // Copy memory from GPU to CPU
    cudaMemcpy((float4 *)hostData, deviceOutData, sizeof(float4) * boidCount, cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA Error Occurred: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // Reformat float4 data as boid_t before returning Image ptr
    copyFloatToBoid((float4 *)hostData, image->data->boids, boidCount);

    return image;
}
