#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "seqBoids.h"

/* Global variables */

int *dim_x;
int *dim_y;
group_t* boid_group;

SeqBoids::SeqBoids() {
    // Initialize local variables
    dim_x = NULL;
    dim_y = NULL;
    boid_group = NULL;
}

SeqBoids::~SeqBoids() {
    // Free data structures
}

/* Boids Code */

static pos_t flock_center_rule(group_t* group, int own_i){
    boid_t* boids = group->boids;
    pos_t center;
    center.x = 0;
    center.y = 0;

    for(int i = 0; i< group->size; i++){
        if(i != own_i){
            center.x += boids[i].position.x;
            center.y += boids[i].position.y;
        }
    }
    
    center.x /= (group->size - 1);
    center.y /= (group->size - 1);
    
    // 0.5% towards the percieved center
    pos_t new_pos;
    new_pos.x = (center.x - boids[own_i].position.x) / 200;
    new_pos.y = (center.y - boids[own_i].position.y) / 200;
    return new_pos;
}

static int dist(boid_t b1, boid_t b2) {
    return abs(b1.position.x - b2.position.x) + abs(b1.position.y - b2.position.y);
}

static pos_t collision_avoidance_rule(group_t* group, int own_i){
    int distance = 10; // Threshold of distance between boids
    boid_t* boids = group->boids;
    pos_t result; 
    result.x = 0;
    result.y = 0;

    for(int i = 0; i< group->size; i++){
        if(i != own_i){
            if(dist(boids[i], boids[own_i]) < distance){
                result.x -= (boids[i].position.x - boids[own_i].position.x);
                result.y -= (boids[i].position.y - boids[own_i].position.y);
            }
        }
    }
    result.x /= 5;
    result.y /= 5;
    return result;
}

static vel_t velocity_matching_rule(group_t* group, int own_i){
    boid_t* boids = group->boids;
    vel_t result; 
    result.x = 0;
    result.y = 0;

    for(int i = 0; i< group->size; i++){
        if(i != own_i){
            result.x += boids[i].velocity.x;
            result.y += boids[i].velocity.y;
        }
    }

    result.x /= (group->size - 1);
    result.y /= (group->size - 1);
    
    // 10% towards the average velocity 
    vel_t new_vel;
    new_vel.x = (result.x - boids[own_i].velocity.x) / 10;
    new_vel.y = (result.y - boids[own_i].velocity.y) / 10;
    return new_vel;

}

static pos_t leader_following_rule(group_t* group, int own_i, pos_t center, int leader){
    boid_t* boids = group->boids;
    pos_t result; 
    result.x = 0;
    result.y = 0;

    int randNum = rand() % (10);
    int upper = 50;
    int lower = -50;

    if(own_i == leader){
        if(randNum < 5){
            result.x += (rand() % (upper - lower + 1)) + lower;
            result.y += (rand() % (upper - lower + 1)) + lower;
        } 
        else{
            result.x += (boids[own_i].position.x - center.x);
            result.y += (boids[own_i].position.y - center.y);
        }
    } 
    else{
        result.x += (boids[leader].position.x - boids[own_i].position.x);
        result.y += (boids[leader].position.y - boids[own_i].position.y);
    }
    result.x /= 10;
    result.y /= 10;
    return result;
}

static vel_t bound_following_rule(group_t* group, int own_i, int Xmin, int Xmax, int Ymin, int Ymax){
    boid_t b = group->boids[own_i];
    vel_t v; 
    v.x = 0;
    v.y = 0;

    int amount = 100;

    if(b.position.x < Xmin)
        v.x = amount;
    else if(b.position.x > Xmax)
        v.x = -1 * amount;

    if(b.position.y < Ymin)
        v.y = amount;
    else if(b.position.y > Ymax)
        v.y = -1 * amount;
    
    return v;
}

static void update_boid_pos(group_t* group, int own_i) {
    boid_t* boids = group->boids;

    int leader = 0;
    int low_x = -100;
    int low_y = -100;
    int high_x = 100;
    int high_y = 100;
    int vlim_x = 20;
    int vlim_y = 20;

    pos_t p1 = flock_center_rule(group, own_i);
    pos_t p2 = collision_avoidance_rule(group, own_i);
    vel_t v3 = velocity_matching_rule(group, own_i);
    pos_t p4 = leader_following_rule(group, own_i, p1, leader);
    vel_t v5 = bound_following_rule(group, own_i, low_x, high_x, low_y, high_y);

    boids[own_i].velocity.x += p1.x + p2.x + v3.x + p4.x + v5.x; 
    boids[own_i].velocity.y += p1.y + p2.y + v3.y + p4.y + v5.y; 

    if(abs(boids[own_i].velocity.x) > vlim_x){
        boids[own_i].velocity.x = (boids[own_i].velocity.x / abs(boids[own_i].velocity.x)) * vlim_x;
    }
    if(abs(boids[own_i].velocity.y) > vlim_y){
        boids[own_i].velocity.y = (boids[own_i].velocity.y / abs(boids[own_i].velocity.y)) * vlim_y;
    }

    // Update the position to the new position
    boids[own_i].position.x += boids[own_i].velocity.x; 
    boids[own_i].position.y += boids[own_i].velocity.y; 
}

/* Update function */
void SeqBoids::updateScene() {
    // For sequential, we just iterate over all boids in order
    int count = boid_group->size;
    for (int i = 0; i < count; i++) {
        update_boid_pos(boid_group, i);
    }
}

/* Input function */
void SeqBoids::setup(const char *inputFile) {
    FILE *input = fopen(inputFile, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", inputFile);
        return;
    }

    dim_x = (int *)calloc(1, sizeof(int));
    dim_y = (int *)calloc(1, sizeof(int));
    int num_of_boids;

    fscanf(input, "%d %d\n", dim_x, dim_y);
    fscanf(input, "%d\n", &num_of_boids);

    // Allocate mem for the boids
    boid_group = (group_t*)malloc(sizeof(group_t));
    boid_t *boids = (boid_t *)calloc(num_of_boids, sizeof(boid_t));
    boid_group->boids = boids;
    boid_group->size = num_of_boids;
    
    /* Read the grid dimension and boid information from file */
    
    // Load the coords (x1,y1) for each boid
    int x1, y1;
    for (int i = 0; i < num_of_boids; i++) {
        fscanf(input, "%d %d\n", &x1, &y1);
        boids[i].position.x = x1;
        boids[i].position.y = y1;
    } 
}

/* Output function */
void writeOutput(group_t* group, int iter, int dim_x, int dim_y){

    char filename[1024];
    sprintf(filename, "./output/%s_%d.txt", "framePositions", iter);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    boid_t* boids = group->boids;
    int num_boids = group->size;

    fprintf(fp, "%d %d\n", dim_x, dim_y);

    for (int i = 0; i < num_boids; i++) {
        fprintf(fp, "%d %d", boids[i].position.x, boids[i].position.y);
        if (i != num_boids - 1)
           fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Wrote boids frame file %s\n", filename);
}

void SeqBoids::output(int iter) {
    writeOutput(boid_group, iter, *dim_x, *dim_y);
}