#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <omp.h>
#include <math.h>
#include <algorithm>

#include "boids.h"
#include "seqBoids.h"

/* Bin Lattice spatial subdivision */
// Represents a link in a singly-linked list, with ptr to boid
struct link_t {
    link_t *next;
    boid_t *boid;
};

// Represents entry in doubly-linked list, with ptr to boid
struct entry_t {
    entry_t *prev;
    entry_t *next;
    boid_t *boid;
};

// A bin is the holder of a linked list
typedef struct {
    entry_t *head;
    entry_t *tail;
} bin_t;

// The lattice grid, bin partition over the simulation space
typedef struct {
    int rows;
    int cols;
    int cellWidth;
    int cellHeight;
    boid_list_t *bins;
} grid_t;

/* Global variables */
Image *image;
group_t *boid_group;
grid_t *grid;
int *gridCoord;

SeqBoids::SeqBoids() {
}

SeqBoids::~SeqBoids() {
}

/* Boids Code */

// Helper function: get absolute distance between two boids
static float dist(boid_t *b1, boid_t *b2) {
    // Uses sqrt((x2-x1)^2 + (y2-y1)^2) calculation
    return sqrt(pow(b1->position.x - b2->position.x, 2) + pow(b1->position.y - b2->position.y, 2));
}


static vel_t leader_steer(boid_t *boid, link_t *flock, boid_t* lead, int flockSize){

    float followFactor = 0.01; // Adjust velocity by this %

    vel_t result; 
    result.x = 0;
    result.y = 0;

    if(boid == lead){
        int move_random = rand() % (10);
        int upper = 50;
        int lower = -50;

        if(move_random < 2){
            result.x += (rand() % (upper - lower + 1)) + lower;
            result.y += (rand() % (upper - lower + 1)) + lower;
        }
        else{
            // First, find the center of our neighborhood (within visibility)
            pos_t center;
            center.x = 0.0f;
            center.y = 0.0f;

            // NOTE: this isn't preventing a potential overflow
            link_t *neighbor = flock;
            while (neighbor != NULL) {
                center.x += neighbor->boid->position.x;
                center.y += neighbor->boid->position.y;
                neighbor = neighbor->next;
            }

            center.x /= (float)flockSize;
            center.y /= (float)flockSize;

            result.x += (center.x - lead->position.x);
            result.y += (center.y - lead->position.y);
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

// static vel_t bound_following_rule(group_t* group, int own_i, int Xmin, int Xmax, int Ymin, int Ymax){
//     boid_t b = group->boids[own_i];
//     vel_t v; 
//     v.x = 0;
//     v.y = 0;

//     int amount = 100;

//     if(b.position.x < Xmin)
//         v.x = amount;
//     else if(b.position.x > Xmax)
//         v.x = -1 * amount;

//     if(b.position.y < Ymin)
//         v.y = amount;
//     else if(b.position.y > Ymax)
//         v.y = -1 * amount;
    
//     return v;
// }

// static void update_boid_pos(group_t* group, int own_i) {
//     boid_t* boids = group->boids;

//     int leader = 0;
//     int low_x = -100;
//     int low_y = -100;
//     int high_x = 100;
//     int high_y = 100;
//     int vlim_x = 20;
//     int vlim_y = 20;

//     pos_t p1 = flock_center_rule(group, own_i);
//     pos_t p2 = collision_avoidance_rule(group, own_i);
//     vel_t v3 = velocity_matching_rule(group, own_i);
//     pos_t p4 = leader_following_rule(group, own_i, p1, leader);
//     vel_t v5 = bound_following_rule(group, own_i, low_x, high_x, low_y, high_y);

//     boids[own_i].velocity.x += p1.x + p2.x + v3.x + p4.x + v5.x; 
//     boids[own_i].velocity.y += p1.y + p2.y + v3.y + p4.y + v5.y; 

//     if(abs(boids[own_i].velocity.x) > vlim_x){
//         boids[own_i].velocity.x = (boids[own_i].velocity.x / abs(boids[own_i].velocity.x)) * vlim_x;
//     }
//     if(abs(boids[own_i].velocity.y) > vlim_y){
//         boids[own_i].velocity.y = (boids[own_i].velocity.y / abs(boids[own_i].velocity.y)) * vlim_y;
//     }

//     // Update the position to the new position
//     boids[own_i].position.x += boids[own_i].velocity.x; 
//     boids[own_i].position.y += boids[own_i].velocity.y; 
// }

vel_t separation_steer(boid_t *boid, link_t *flock, float d) {
    float minDistance = 40.0; // The distance to stay away from other boids
    float avoidFactor = 8.0; // Adjust velocity by this %

    // s_i = -Sum over neighbors s_j: (p_i - p_j)
    vel_t result;
    result.x = 0.0f;
    result.y = 0.0f;

    link_t *neighbor = flock;
    while (neighbor != NULL) {
        if (dist(neighbor->boid, boid) <= minDistance) {     
            result.x += (boid->position.x - neighbor->boid->position.x);
            result.y += (boid->position.y - neighbor->boid->position.y);
        }
        neighbor = neighbor->next;
    }

    result.x = result.x * avoidFactor;
    result.y = result.y * avoidFactor;

    // printf("separation: %4f, %4f\n", result.x, result.y);
    return result;
}

vel_t cohesion_steer(boid_t *boid, link_t *flock, int flockSize) {

    float centeringFactor = 0.005; // adjust velocity by this %

    // First, find the center of our neighborhood (within visibility)
    pos_t center;
    center.x = 0.0f;
    center.y = 0.0f;

    // NOTE: this isn't preventing a potential overflow
    link_t *neighbor = flock;
    while (neighbor != NULL) {
        center.x += neighbor->boid->position.x;
        center.y += neighbor->boid->position.y;
        neighbor = neighbor->next;
    }

    center.x /= (float)flockSize;
    center.y /= (float)flockSize;

    // Second, calculate the cohesion displacement vector
    vel_t result;
    result.x = (center.x - boid->position.x) * centeringFactor; 
    result.y = (center.y - boid->position.y) * centeringFactor;

    // printf("cohesion:  %4f, %4f\n", result.x, result.y);
    return result;
}

vel_t alignment_steer(boid_t *boid, link_t *flock, int flockSize) {

    float matchingFactor = 0.05; // Adjust by this % of average velocity

    vel_t result;
    result.x = 0.0f;
    result.y = 0.0f;

    // NOTE: this isn't preventing a potential overflow
    link_t *neighbor = flock;
    while (neighbor != NULL) {
        result.x += neighbor->boid->velocity.x;
        result.y += neighbor->boid->velocity.y;
        neighbor = neighbor->next;
    }

    result.x /= (float)flockSize;
    result.y /= (float)flockSize;

    result.x = (result.x - boid->velocity.x) * matchingFactor;
    result.y = (result.y - boid->velocity.y) * matchingFactor;

    // printf("alignment: %4f, %4f\n", result.x, result.y);
    return result;
}

void update_boid(int i, float deltaT, float e, float s, float k, float m, float l, Image *image, boid_t* lead) {
    // First, find the visible neighborhood of this boid
    link_t *flock = NULL;
    int flockSize = 0;

    // TODO: Rather than alloc the neighborhood, accumulate the data needed for force calcs
    // as you go, i.e. copy the position/velocity of that boid and accumulate for later.

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

    for (int x = gridMinX; x <= gridMaxX; x++) {
        for (int y = gridMinY; y <= gridMaxY; y++) {
            // Check all the boids in this bin to see if within radius
            boid_list_t* bin = &grid->bins[x * grid->cols + y];

            boid_t* boid = Q_GET_FRONT(bin);
            while(boid != NULL){
                if (boid != myBoid && dist(boid, myBoid) <= e) {
                    link_t *newNeighbor = (link_t *)malloc(sizeof(link_t));
                    newNeighbor->boid = boid;
                    newNeighbor->next = flock;
                    flock = newNeighbor;
                    flockSize++;
                }
                boid = Q_GET_NEXT(boid, grid_link);
            }
        }
    }

    // Calculate the separation, cohesion, and alignment steers
    vel_t vel_i = myBoid->velocity;
    
    // printf("boid %d, flocksize %d\n", i, flockSize);
    if (flockSize > 0) {
        vel_t sep_i = separation_steer(myBoid, flock, e / 2.0f);
        vel_t coh_i = cohesion_steer(myBoid, flock, flockSize);
        vel_t ali_i = alignment_steer(myBoid, flock, flockSize);
        vel_t lead_i = leader_steer(myBoid, flock, lead, flockSize);

        vel_i.x += (s * sep_i.x) + (k * coh_i.x) + (m * ali_i.x) + (l * lead_i.x);
        vel_i.y += (s * sep_i.y) + (k * coh_i.y) + (m * ali_i.y) + (l * lead_i.y); 
    }
    else{
        vel_i.x += (rand() % (2 - 0 + 1)) + 0;
        vel_i.y +=(rand() % (2 - 0 + 1)) + 0;
    }

    // printf("vel new: %4f, %4f\n", vel_i.x, vel_i.y);

    // Cap the possible velocity
    // float vlim = 5.0f;
    // if (abs(vel_i.x) > vlim) {
    //     vel_i.x = (vel_i.x / abs(vel_i.x)) * vlim;
    // }
    // if (abs(vel_i.y) > vlim) {
    //     vel_i.y = (vel_i.y / abs(vel_i.y)) * vlim;
    // }
    float speedLimit = 15.0;
    float speed = sqrt(vel_i.x * vel_i.x + vel_i.y * vel_i.y);
    if (speed > speedLimit) {
        vel_i.x = (vel_i.x  / speed) * speedLimit;
        vel_i.y = (vel_i.y / speed) * speedLimit;
    }

    // printf("vel new: %4f, %4f\n", vel_i.x, vel_i.y);

    // Update the position and velocity of this boid
    myBoid->velocity = vel_i;
    float newPosX = myBoid->position.x + deltaT * myBoid->velocity.x;
    float newPosY = myBoid->position.y + deltaT * myBoid->velocity.y;

    // Check whether the boid's new position is in bounds
    // For now: wrap around to other side of the image
    if (newPosX + (image->width / 2) < 0) {
        newPosX = image->width + newPosX;
        // printf("Boid went out of bounds, moved to x %lf.\n", newPosX);
    }
    else if (newPosX > (image->width / 2)) {
        newPosX = newPosX - image->width;
        // printf("Boid went out of bounds, moved to x %lf.\n", newPosX);
    }
    if (newPosY + (image->height / 2) < 0) {
        newPosY = image->height + newPosY;
        // printf("Boid went out of bounds, moved to y %lf.\n", newPosY);
    }
    else if (newPosY > (image->height / 2)) {
        newPosY = newPosY - image->height;
        // printf("Boid went out of bounds, moved to y %lf.\n", newPosY);
    }
    // const margin = 10;
    // const turnFactor = 1;

    // if (boid.x < margin) {
    //     boid.dx += turnFactor;
    // }
    // if (boid.x > width - margin) {
    //     boid.dx -= turnFactor
    // }
    // if (boid.y < margin) {
    //     boid.dy += turnFactor;
    // }
    // if (boid.y > height - margin) {
    //     boid.dy -= turnFactor;
    // }

    myBoid->position.x = newPosX;
    myBoid->position.y = newPosY;
}

/* To prevent the data structure from being modified during execution,
 * update the grid at the end of the frame update based on new positions. */
void update_grid (Image *image) {
    int boidCount = boid_group->count;
    boid_t *boids = boid_group->boids;

    for (int i = 0; i < boidCount; i++) {
        // Find which grid cell bin this boid should be in
        int gridX = (boids[i].position.x + (image->width / 2)) / grid->cellWidth;
        int gridY = (boids[i].position.y + (image->height / 2)) / grid->cellHeight;

        // Check if the grid position has changed since prev frame
        int oldGridX = boids[i].grid_x;
        int oldGridY = boids[i].grid_y;

        if (oldGridX == gridX && oldGridY == gridY)
            continue;

        boids[i].grid_x = gridX;
        boids[i].grid_y = gridY;

        // Remove from the old bin data structure
        boid_list_t* oldBin = &grid->bins[oldGridX * grid->cols + oldGridY];
        Q_REMOVE(oldBin, &boids[i], grid_link);

        // Add to the new bin data structure
        boid_list_t* newBin = &grid->bins[gridX * grid->cols + gridY];
        Q_INIT_ELEM(&boids[i], grid_link);
        Q_INSERT_TAIL(newBin, &boids[i], grid_link);
    }

    // for(int r = 0; r < grid->rows; r++){
    //     for(int c = 0; c < grid->cols; c++){
    //         printf("gridx: %d, gridy: %d\n", r,c);
    //         boid_list_t* bin = &(grid->bins[r * grid->cols + c]);
    //         boid_t* boid = Q_GET_FRONT(bin);
    //         while(boid != NULL){
    //             printf("boid %d\n", boid->index);
    //             boid = Q_GET_NEXT(boid, grid_link);
    //         }
    //     }
    // }
}

///////////////////////////////////////////////////////////////////////////////

/* Update function */
void SeqBoids::updateScene() {
    // Parameters: need to make these modifiable outside of code
    // float deltaT = 0.5f;
    // float e = 75.0f;
    // float s = -1.0f;
    // float k = 0.8f;
    // float m = 0.8f;
    // float l = 0.8f;
    float deltaT = 1.0f;
    float e = 75.0f;
    float s = 1.0f;
    float k = 1.0f;
    float m = 1.0f;
    float l = 1.0f;

    int leader = 0;
    boid_t* lead = &(boid_group->boids[leader]);

    // For sequential, we just iterate over all boids in order
    int boidCount = boid_group->count;
    int i;
    //#pragma omp parallel for default(shared) private(i) shared(costs) schedule(dynamic)
    for (i = 0; i < boidCount; i++) {
        update_boid(i, deltaT, e, s, k, m, l, image, lead);
    }

    // After that's finished, update the data structure
    update_grid(image);
}

/* Input function */
void SeqBoids::setup(const char *inputFile, int num_of_threads) {

    //omp_set_num_threads(num_of_threads);

    FILE *input = fopen(inputFile, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", inputFile);
        return;
    }

    int dim_x;
    int dim_y;
    fscanf(input, "%d %d\n", &dim_x, &dim_y);

    printf("Allocating image of dim %d %d.\n", dim_x, dim_y);
  
    // Note: input dim_x=500 dim_y=500 implies image of dim [-500,500],[-500,500].
    image = (Image *)malloc(sizeof(Image));
    image->width = 2 * dim_x;
    image->height = 2 * dim_y;

    printf("Image width %d and height %d.\n", image->width, image->height);

    int num_of_boids;
    fscanf(input, "%d\n", &num_of_boids);

    // Allocate mem for the boids
    image->data = (group_t*)malloc(sizeof(group_t));
    
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

    // TODO: Establish parameters/test how the grid should be split
   
    // Initialize the spatial partitioning data structure
    grid = (grid_t*)malloc(sizeof(grid_t));
    grid->cellWidth = 10;
    grid->cellHeight = 10;
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
    // gridCoord = (int *)calloc(2 * num_of_boids, sizeof(int));
 
    for (int i = 0; i < num_of_boids; i++) {
        // Find which grid cell bin this boid should be in
        boids[i].grid_x = (boids[i].position.x + (image->width / 2)) / grid->cellWidth;
        boids[i].grid_y = (boids[i].position.y + (image->height / 2)) / grid->cellHeight;

        int gridX = boids[i].grid_x;
        int gridY = boids[i].grid_y;

        // printf("boid %d, gridx: %d gridy: %d\n", i, gridX, gridY);

        // Add to the bin data structure
        boid_list_t* bin = &grid->bins[gridX * grid->cols + gridY];
        Q_INIT_ELEM(&boids[i], grid_link);
        Q_INSERT_TAIL(bin, &boids[i], grid_link);

        assert(NULL != Q_GET_FRONT(bin));
        assert(&(boids[i]) == Q_GET_TAIL(bin));
        assert(Q_GET_TAIL(bin) -> index == i);
    }

    // for(int r = 0; r < grid->rows; r++){
    //     for(int c = 0; c < grid->cols; c++){
    //         printf("gridx: %d, gridy: %d\n", r,c);
    //         boid_list_t* bin = &(grid->bins[r * grid->cols + c]);
    //         boid_t* boid = Q_GET_FRONT(bin);
    //         while(boid != NULL){
    //             printf("boid %d\n", boid->index);
    //             boid = Q_GET_NEXT(boid, grid_link);
    //         }
    //     }
    // }
}

/* Output function */
Image *SeqBoids::output() {
    // Already the data that we're operating on
    return image;
}
