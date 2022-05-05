#ifndef __OPENMP_BOIDS_H__
#define __OPENMP_BOIDS_H__

#include "boids.h"

class OpenmpBoids : public Boids {

  private:
    Image *image;
    group_t *boid_group;
    grid_t *grid;
    int *gridCoord;
    int total_threads;

  public:
    OpenmpBoids();

    virtual ~OpenmpBoids();
    
    void setup(const char* inputName, int num_of_threads);

    void updateScene();

    Image *output();

    // To avoid conflicts between the cpu implementations
    vel_t leader_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, boid_t *lead, int flockSize);
    vel_t separation_steer(boid_t *boid, pos_t p_sum, float d, int flockSize);
    vel_t cohesion_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, int flockSize);
    vel_t alignment_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, int flockSize);
    void update_boid(int i, float deltaT, float e, float s, float k, float m, float l, boid_t *lead);
    void update_grid();
};

#endif
