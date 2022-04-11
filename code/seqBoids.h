#ifndef __SEQ_BOIDS_H__
#define __SEQ_BOIDS_H__

#include "boids.h"

class SeqBoids : public Boids {
  /*
  private:
    Image *image;
    SceneName sceneName;

    int numberOfCircles;
    float *position;
    float *velocity;
    float *color;
    float *radius;
  */
  public:
    SeqBoids();
    virtual ~SeqBoids();
    
    //const Image *getImage();

    void setup(const char* inputName);

    //void loadScene(SceneName name);

    //void allocOutputImage(int width, int height);

    //void clearImage();

    void updateScene();

    //void render();
    
    void output(int frame);

    //void dumpParticles(const char *filename);

    //void shadePixel(float pixelCenterX, float pixelCenterY, float px, float py, float pz,
    //                float *pixelData, int circleIndex);
};

#endif
