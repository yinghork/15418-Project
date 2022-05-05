#ifndef __CUDA_BOIDS_H__
#define __CUDA_BOIDS_H__

#ifndef uint
#define uint unsigned int
#endif

#include "boids.h"

class CudaBoids : public Boids {

  private:
    Image *image;
    float4 *hostData;
    int boidCount;
    float4 *deviceInData;
    float4 *deviceOutData;
 
  public:
    CudaBoids();
    virtual ~CudaBoids();
    
    void setup(const char* inputName, int num_of_threads);

    void updateScene();

    Image *output();
};

#endif
