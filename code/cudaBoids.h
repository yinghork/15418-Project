#ifndef __CUDA_BOIDS_H__
#define __CUDA_BOIDS_H__

#ifndef uint
#define uint unsigned int
#endif

#include "boids.h"

class CudaBoids : public Boids {

  private:
    Image *image; 
    int boidCount;
    boid_t *cudaDeviceBoidData;

  /*
    SceneName sceneName;

    int numberOfCircles;
    float *position;
    float *velocity;
    float *color;
    float *radius;

    float *cudaDevicePosition;
    float *cudaDeviceVelocity;
    float *cudaDeviceColor;
    float *cudaDeviceRadius;
    float *cudaDeviceImageData;
  */
  public:
    CudaBoids();
    virtual ~CudaBoids();
    
    void setup(const char* inputName);

    void updateScene();

    Image *output();

    //const Image *getImage();

    //void setup();

    //void loadScene(SceneName name);

    //void allocOutputImage(int width, int height);

    //void clearImage();

    //void advanceAnimation();

    //void shadePixel(float pixelCenterX, float pixelCenterY, float px, float py, float pz,
    //                float *pixelData, int circleIndex);
};

#endif
