#ifndef __BOIDS_H__
#define __BOIDS_H__

// struct Image;

// fireworks constants
//#define NUM_FIREWORKS 15
//#define NUM_SPARKS 20

/*
typedef enum {
    CIRCLE_RGB,
    CIRCLE_RGBY,
    CIRCLE_TEST_10K,
    CIRCLE_TEST_100K,
    PATTERN,
    SNOWFLAKES,
    BOUNCING_BALLS,
    HYPNOSIS,
    FIREWORKS,
    SNOWFLAKES_SINGLE_FRAME,
    BIG_LITTLE,
    LITTLE_BIG
} SceneName;
*/

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

typedef struct {
    pos_t position;
    vel_t velocity;
} boid_t;

typedef struct {
    int count;
    boid_t *boids;
} group_t;

/* Struct which defines the data structure which holds output */
struct Image {

    Image(int w, int h) {
        width = w;
        height = h;
        data = NULL;
    }

    int width;
    int height;
    group_t *data;
};

class Boids {
  public:
    virtual ~Boids(){};

    virtual void setup(const char *inputName) = 0;

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
