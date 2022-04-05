/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <omp.h>
#include <random>

typedef struct { 
    int x;
    int y;
} pos_t;

typedef struct { 
    int x;
    int y;
} vel_t;

typedef struct { 
    pos_t position;
    vel_t velocity;
} boid_t;

typedef struct { 
    boid_t* boids;
    int size = 0;
} group_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif
