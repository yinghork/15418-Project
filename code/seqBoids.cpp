#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <math.h>
#include <algorithm>

#include "boids.h"
#include "seqBoids.h"

/* Class variables (defined in header file) */
//Image *image;
//group_t *boid_group;
//grid_t *grid;
//int *gridCoord;
//int total_threads; 

/* Boids Code */

// Helper function: get absolute distance between two boids
static float dist(boid_t *b1, boid_t *b2) {
    return sqrt(pow(b1->position.x - b2->position.x, 2) + pow(b1->position.y - b2->position.y, 2));
}


vel_t SeqBoids::leader_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, boid_t* lead, int flockSize){

    float followFactor = 0.001; // Adjust velocity by this %

    vel_t result; 
    result.x = 0;
    result.y = 0;

    if(boid == lead){
        int move_random = rand() % (10);
        int upper = 1000;
        int lower = -1000;

        if(move_random < 2){
            result.x += (rand() % (upper - lower + 1)) + lower;
            result.y += (rand() % (upper - lower + 1)) + lower;
        }
    } 
    else{
        result.x += (lead->position.x - boid->position.x);
        result.y += (lead->position.y - boid->position.y);
    }
    result.x *= followFactor;
    result.y *= followFactor;
    return result;
}

vel_t SeqBoids::separation_steer(boid_t *boid, pos_t p_sum, float d, int flockSize) {
    float avoidFactor = 10.0;

    vel_t result;
    result.x = flockSize * boid->position.x - p_sum.x;
    result.y = flockSize * boid->position.y - p_sum.y;

    result.x = result.x * avoidFactor;
    result.y = result.y * avoidFactor;

    return result;
}

vel_t SeqBoids::cohesion_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, int flockSize) {

    float centeringFactor = 0.005; // adjust velocity by this %

    // First, find the center of our neighborhood (within visibility)
    pos_t center;
    center.x = p_sum.x;
    center.y = p_sum.y;

    center.x /= (float)flockSize;
    center.y /= (float)flockSize;

    // Second, calculate the cohesion displacement vector
    vel_t result;
    result.x = (center.x - boid->position.x) * centeringFactor; 
    result.y = (center.y - boid->position.y) * centeringFactor;

    return result;
}

vel_t SeqBoids::alignment_steer(boid_t *boid, vel_t v_sum, pos_t p_sum, int flockSize) {

    float matchingFactor = 0.05; // Adjust by this % of average velocity

    vel_t result;
    result.x = v_sum.x;
    result.y = v_sum.y;

    result.x /= (float)flockSize;
    result.y /= (float)flockSize;

    result.x = (result.x - boid->velocity.x) * matchingFactor;
    result.y = (result.y - boid->velocity.y) * matchingFactor;

    return result;
}

void SeqBoids::update_boid(int i, float deltaT, float e, float s, float k, float m, float l, boid_t* lead) {

    float minDistance = 40.0; // The distance to stay away from other boids

    // First, find the visible neighborhood of this boid
    int flockSize = 0;
    int neighSize = 0;
    group_t *boid_group = image->data;
    boid_t *myBoid = &(boid_group->boids[i]);
    int gridX = myBoid->grid_x;
    int gridY = myBoid->grid_y;

    // Determine which nearby bins we need to check, based on radius of vis (e)
    int visGridX = (e + grid->cellWidth - 1) / grid->cellWidth;
    int visGridY = (e + grid->cellHeight - 1) / grid->cellHeight;

    int gridMinX = std::max(gridX - visGridX, 0);
    int gridMaxX = std::min(gridX + visGridX, grid->cols - 1);
    int gridMinY = std::max(gridY - visGridY, 0);
    int gridMaxY = std::min(gridY + visGridY, grid->rows - 1);

    vel_t v_sum;
    v_sum.x = 0;
    v_sum.y = 0;
    pos_t p_sum;
    p_sum.x = 0;
    p_sum.y = 0;

    pos_t neigh_p_sum;
    neigh_p_sum.x = 0;
    neigh_p_sum.y = 0;

    for (int x = gridMinX; x <= gridMaxX; x++) {
        for (int y = gridMinY; y <= gridMaxY; y++) {
            // Check all the boids in this bin to see if within radius
            boid_list_t* bin = &grid->bins[x * grid->cols + y];

            boid_t* boid = Q_GET_FRONT(bin);
            while(boid != NULL){
                float d = dist(boid, myBoid);
                if (boid != myBoid && d <= e) {

                    vel_t v = boid->velocity;
                    pos_t p = boid->position;
                    
                    v_sum.x += v.x;
                    v_sum.y += v.y;
                    p_sum.x += p.x;
                    p_sum.y += p.y;

                    flockSize++;

                    if(d <= minDistance) {
                    
                        neigh_p_sum.x += p.x;
                        neigh_p_sum.y += p.y;

                        neighSize++;
                    }
                }
                boid = Q_GET_NEXT(boid, grid_link);
            }
        }
    }

    // Calculate the separation, cohesion, and alignment steers
    vel_t vel_i = myBoid->velocity;
    
    if (flockSize > 0) {
        vel_t sep_i = separation_steer(myBoid, neigh_p_sum, e / 2.0f, neighSize);
        vel_t coh_i = cohesion_steer(myBoid, v_sum, p_sum, flockSize);
        vel_t ali_i = alignment_steer(myBoid, v_sum, p_sum, flockSize);
        vel_t lead_i = leader_steer(myBoid, v_sum, p_sum, lead, flockSize);

        vel_i.x += (s * sep_i.x) + (k * coh_i.x) + (m * ali_i.x) + (l * lead_i.x);
        vel_i.y += (s * sep_i.y) + (k * coh_i.y) + (m * ali_i.y) + (l * lead_i.y); 
    }
    else{
        vel_i.x += (rand() % (2 - 0 + 1)) + 0;
        vel_i.y +=(rand() % (2 - 0 + 1)) + 0;
    }

    float speedLimit = 15.0;
    float speed = sqrt(vel_i.x * vel_i.x + vel_i.y * vel_i.y);
    if (speed > speedLimit) {
        vel_i.x = (vel_i.x  / speed) * speedLimit;
        vel_i.y = (vel_i.y / speed) * speedLimit;
    }

    // Update the position and velocity of this boid
    myBoid->velocity = vel_i;
    float newPosX = myBoid->position.x + deltaT * myBoid->velocity.x;
    float newPosY = myBoid->position.y + deltaT * myBoid->velocity.y;

    // Check whether the boid's new position is in bounds
    int margin = 200;
    int turnFactor = 8;

    if (myBoid->position.x + (image->width / 2) < margin) {
        myBoid->velocity.x += turnFactor;
    }
    if (myBoid->position.x > (image->width / 2) - margin) {
        myBoid->velocity.x -= turnFactor;
    }
    if (myBoid->position.y + (image->height / 2) < margin) {
        myBoid->velocity.y += turnFactor;
    }
    if (myBoid->position.y > (image->height / 2) - margin) {
        myBoid->velocity.y -= turnFactor;
    }

    myBoid->position.x = newPosX;
    myBoid->position.y = newPosY;

}

/* To prevent the data structure from being modified during execution,
 * update the grid at the end of the frame update based on new positions. */
void SeqBoids::update_grid () {
    group_t *boid_group = image->data;
    int boidCount = boid_group->count;
    boid_t *boids = boid_group->boids;

    for (int i = 0; i < boidCount; i++) {
        // Find which grid cell bin this boid should be in
        int gridX = (boids[i].position.x + (image->width / 2)) / grid->cellWidth;
        int gridY = (boids[i].position.y + (image->height / 2)) / grid->cellHeight;

        // Check if the grid position has changed since prev frame
        int oldGridX = boids[i].grid_x;
        int oldGridY = boids[i].grid_y;

        if ((oldGridX != gridX || oldGridY != gridY) && gridX >= 0 && gridY >= 0 
                && gridX < grid->rows && gridY < grid->cols) {

            boids[i].grid_x = gridX;
            boids[i].grid_y = gridY;

            boid_list_t* oldBin = &grid->bins[oldGridX * grid->cols + oldGridY];
            boid_list_t* newBin = &grid->bins[gridX * grid->cols + gridY];

            // Remove from the old bin data structure
            Q_REMOVE(oldBin, &boids[i], grid_link);

            // Add to the new bin data structure
            Q_INIT_ELEM(&boids[i], grid_link);
            Q_INSERT_TAIL(newBin, &boids[i], grid_link);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

SeqBoids::SeqBoids() {
    image = NULL;
    boid_group = NULL;
    grid = NULL;
    gridCoord = NULL;
}

SeqBoids::~SeqBoids() {
    if (image) {
        delete gridCoord;
        delete grid;
        delete boid_group;
        delete image;
    }
}

/* Update function */
void SeqBoids::updateScene() {

    // Parameters
    float deltaT = 1.0f;
    float e = 200.0f;
    float s = 1.0f;
    float k = 1.0f;
    float m = 1.0f;
    float l = 0.0f;

    int leader = 0;
    boid_t* lead = &(boid_group->boids[leader]);

    // We iterate over grid sequentially
    for (int i = 0; i < grid->rows * grid->cols; i++) {
        boid_list_t* bin = &grid->bins[i];

        boid_t* boid = Q_GET_FRONT(bin);
        while(boid != NULL){
            update_boid(boid->index, deltaT, e, s, k, m, l, lead);
            boid = Q_GET_NEXT(boid, grid_link);
        }   
    }

    // After that's finished, update the data structure
    update_grid();
}


/* Input function */
void SeqBoids::setup(const char *inputFile, int num_of_threads) {

    omp_set_num_threads(num_of_threads);

    FILE *input = fopen(inputFile, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", inputFile);
        return;
    }

    int dim_x;
    int dim_y;
    fscanf(input, "%d %d\n", &dim_x, &dim_y);

    // printf("Allocating image of dim %d %d.\n", dim_x, dim_y);
  
    // Note: input dim_x=500 dim_y=500 implies image of dim [-500,500],[-500,500].
    image = (Image *)malloc(sizeof(Image));
    if(image == NULL){
        printf("malloc failed");
    }
    image->width = 2 * dim_x;
    image->height = 2 * dim_y;

    // printf("Image width %d and height %d.\n", image->width, image->height);

    int num_of_boids;
    fscanf(input, "%d\n", &num_of_boids);

    // Allocate mem for the boids
    image->data = (group_t*)malloc(sizeof(group_t));
    if(image->data == NULL){
        printf("malloc failed");
    }
    
    boid_group = image->data;
    boid_group->count = num_of_boids;
    
    boid_t *boids = (boid_t *)calloc(num_of_boids, sizeof(boid_t));
    boid_group->boids = boids;
    
    /* Read the grid dimension and boid information from file */
    
    // Load the coords (x1,y1) for each boid
    int x1, y1;
    for (int i = 0; i < num_of_boids; i++) {
        fscanf(input, "%d %d\n", &x1, &y1);
        boids[i].position.x = (float)x1;
        boids[i].position.y = (float)y1;
        boids[i].index = i;
    }
   
    // Initialize the spatial partitioning data structure
    grid = (grid_t*)malloc(sizeof(grid_t));
    
    grid->cellWidth = 2;
    grid->cellHeight = 2;
    grid->rows = image->height / grid->cellHeight;
    grid->cols = image->width / grid->cellWidth;
    // ASSERT: width and height are evenly divided into cells
    grid->bins = (boid_list_t*)malloc(grid->rows * grid->cols * sizeof(boid_list_t));

    for(int r = 0; r < grid->rows; r++){
        for(int c = 0; c < grid->cols; c++){
            Q_INIT_HEAD(&grid->bins[r * grid->cols + c]);
        }
    }
   
    // Assign each boid to its initial bin
    // For now, have an array of grid ID that a boid can check,
    // since it's not currently represented in the boid data struct.
 
    for (int i = 0; i < num_of_boids; i++) {
        // Find which grid cell bin this boid should be in
        boids[i].grid_x = (boids[i].position.x + (image->width / 2)) / grid->cellWidth;
        boids[i].grid_y = (boids[i].position.y + (image->height / 2)) / grid->cellHeight;

        int gridX = boids[i].grid_x;
        int gridY = boids[i].grid_y;

        // Add to the bin data structure
        boid_list_t* bin = &grid->bins[gridX * grid->cols + gridY];
        Q_INIT_ELEM(&boids[i], grid_link);
        Q_INSERT_TAIL(bin, &boids[i], grid_link);

        assert(NULL != Q_GET_FRONT(bin));
        assert(&(boids[i]) == Q_GET_TAIL(bin));
        assert(Q_GET_TAIL(bin) -> index == i);
    }
}

/* Output function */
Image *SeqBoids::output() {
    // Already the data that we're operating on
    return image;
}
