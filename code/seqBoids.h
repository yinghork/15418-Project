#ifndef __SEQ_BOIDS_H__
#define __SEQ_BOIDS_H__

#include "boids.h"

class SeqBoids : public Boids {

  private:
    Image *image;
    /*
    int numberOfCircles;
    float *position;
    float *velocity;
    float *color;
    float *radius;
    */

  public:
    SeqBoids();

    virtual ~SeqBoids();
    
    void setup(const char* inputName);

    void updateScene();

    Image *output();
    
    //void loadScene(SceneName name);

    //void allocOutputImage(int width, int height);

    //void clearImage();

    //void render();
    
    //void dumpParticles(const char *filename);

    //void shadePixel(float pixelCenterX, float pixelCenterY, float px, float py, float pz,
    //                float *pixelData, int circleIndex);
};

#endif
