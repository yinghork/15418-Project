#include "variable_queue.h"

#ifndef __BOIDS_H__
#define __BOIDS_H__

// list of boids
Q_NEW_HEAD(boid_list_t, boid);

/* Structs used to define the boids.
 * A boid consists of its position and velocity in 2d space. */
typedef struct {
    float x;
    float y;
} pos_t;

typedef struct {
    float x;
    float y;
} vel_t;

typedef struct boid {
    pos_t position;
    vel_t velocity;

    int index;
    int grid_x;
    int grid_y;
    Q_NEW_LINK(boid) grid_link;

} boid_t;

typedef struct {
    int count;
    boid_t *boids;
} group_t;

/* Struct which defines the data structure which holds output */
struct Image {
    int width;
    int height;
    group_t *data;
};

/* Bin Lattice spatial subdivision */
// The lattice grid, bin partition over the simulation space
typedef struct {
    int rows;
    int cols;
    int cellWidth;
    int cellHeight;
    boid_list_t *bins;
} grid_t;

class Boids {
  public:
    virtual ~Boids(){};

    virtual void setup(const char *inputName, int num_of_threads) = 0;

    virtual void updateScene() = 0;

    virtual Image *output() = 0;

    //virtual const Image *getImage() = 0;

    //virtual void setup() = 0;

    //virtual void loadScene(SceneName name) = 0;

    //virtual void allocOutputImage(int width, int height) = 0;

    //virtual void clearImage() = 0;

    //virtual void advanceAnimation() = 0;

    //virtual void render() = 0;

    // virtual void dumpParticles(const char* filename) {}
};

#endif
