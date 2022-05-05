#ifndef __CUDA_BOIDS_H__
#define __CUDA_BOIDS_H__

#ifndef uint
#define uint unsigned int
#endif

#include "boids.h"

class CudaBoids : public Boids {

  private:
    Image *image;
    float *hostData;
    int boidCount;
    float *deviceInData;
    float *deviceOutData;
 
  public:
    CudaBoids();
    virtual ~CudaBoids();
    
    void setup(const char* inputName, int num_of_threads);

    void updateScene();

    Image *output();
};

#endif
