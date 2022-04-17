#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <math.h>
#include <algorithm>

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
    bin_t *bins;
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
    
    // Uses manhattan distance, (x2-x1)+(y2-y1).
    // return abs(b1.position.x - b2.position.x) + abs(b1.position.y - b2.position.y);
}

static pos_t flock_center_rule(group_t* group, int own_i){
    boid_t* boids = group->boids;
    pos_t center;
    center.x = 0;
    center.y = 0;

    for(int i = 0; i< group->count; i++){
        if(i != own_i){
            center.x += boids[i].position.x;
            center.y += boids[i].position.y;
        }
    }
    
    center.x /= (group->count - 1);
    center.y /= (group->count - 1);
    
    // 0.5% towards the percieved center
    pos_t new_pos;
    new_pos.x = (center.x - boids[own_i].position.x) / 200;
    new_pos.y = (center.y - boids[own_i].position.y) / 200;
    return new_pos;
}

static pos_t collision_avoidance_rule(group_t* group, int own_i){
    int distance = 10; // Threshold of distance between boids
    boid_t* boids = group->boids;
    pos_t result; 
    result.x = 0;
    result.y = 0;

    for(int i = 0; i< group->count; i++){
        if(i != own_i){
            if(dist(&boids[i], &boids[own_i]) < distance){
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

    for(int i = 0; i< group->count; i++){
        if(i != own_i){
            result.x += boids[i].velocity.x;
            result.y += boids[i].velocity.y;
        }
    }

    result.x /= (group->count - 1);
    result.y /= (group->count - 1);
    
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

vel_t separation_steer(boid_t *boid, link_t *flock, float d) {
    // s_i = -Sum over neighbors s_j: (p_i - p_j)
    vel_t result;
    result.x = 0.0f;
    result.y = 0.0f;

    link_t *neighbor = flock;
    while (neighbor != NULL) {
        if (dist(neighbor->boid, boid) <= d) {     
            result.x -= boid->position.x - neighbor->boid->position.x;
            result.y -= boid->position.y - neighbor->boid->position.y;
        }
        neighbor = neighbor->next;
    }
    
    return result;
}

vel_t cohesion_steer(boid_t *boid, link_t *flock, int flockSize) {
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
    result.x = center.x - boid->position.x;
    result.y = center.y - boid->position.y;

    return result;
}

vel_t alignment_steer(boid_t *boid, link_t *flock, int flockSize) {
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

    return result;
}

void update_boid(int i, float deltaT, float e, float s, float k, float m, Image *image) {
    // First, find the visible neighborhood of this boid
    link_t *flock = NULL;
    int flockSize = 0;

    // TODO: Rather than alloc the neighborhood, accumulate the data needed for force calcs
    // as you go, i.e. copy the position/velocity of that boid and accumulate for later.

    boid_t *myBoid = &(boid_group->boids[i]);
    int gridX = gridCoord[2*i];
    int gridY = gridCoord[2*i+1];

    // Determine which nearby bins we need to check, based on radius of vis (e)
    int visGridX = e / grid->cellWidth;
    int visGridY = e / grid->cellHeight;

    int gridMinX = std::max(gridX - visGridX, 0);
    int gridMaxX = std::min(gridX + visGridX, grid->cols - 1);
    int gridMinY = std::max(gridY - visGridY, 0);
    int gridMaxY = std::min(gridY + visGridY, grid->rows - 1);

    for (int x = gridMinX; x <= gridMaxX; x++) {
        for (int y = gridMinY; y <= gridMaxY; y++) {
            // Check all the boids in this bin to see if within radius
            bin_t *bin = &(grid->bins[x * grid->cols + y]);
            entry_t *entry = bin->head;
            while (entry != NULL) {
                if (entry->boid != myBoid && dist(entry->boid, myBoid) <= e) {
                    link_t *newNeighbor = (link_t *)malloc(sizeof(link_t));
                    newNeighbor->boid = entry->boid;
                    newNeighbor->next = flock;
                    flock = newNeighbor;
                    flockSize++;
                }
                entry = entry->next;
            }
        }
    }

    // Calculate the separation, cohesion, and alignment steers
    vel_t vel_i = myBoid->velocity;
    
    if (flockSize > 0) {
        vel_t sep_i = separation_steer(myBoid, flock, e / 2.0f);
        vel_t coh_i = cohesion_steer(myBoid, flock, flockSize);
        vel_t ali_i = alignment_steer(myBoid, flock, flockSize);
        vel_i.x += (s * sep_i.x) + (k * coh_i.x) + (m * ali_i.x);
        vel_i.y += (s * sep_i.y) + (k * coh_i.y) + (m * ali_i.y);
    }

    // Cap the possible velocity
    float vlim = 20.0f;
    if (abs(vel_i.x) > vlim) {
        vel_i.x = (vel_i.x / abs(vel_i.x)) * vlim;
    }
    if (abs(vel_i.y) > vlim) {
        vel_i.y = (vel_i.y / abs(vel_i.y)) * vlim;
    }

    // Update the position and velocity of this boid
    myBoid->velocity = vel_i;
    float newPosX = myBoid->position.x + deltaT * myBoid->velocity.x;
    float newPosY = myBoid->position.y + deltaT * myBoid->velocity.y;

    // Check whether the boid's new position is in bounds
    // For now: wrap around to other side of the image
    if (newPosX + (image->width / 2) < 0) {
        newPosX = image->width + newPosX;
        printf("Boid went out of bounds, moved to x %lf.\n", newPosX);
    }
    else if (newPosX > (image->width / 2)) {
        newPosX = newPosX - image->width;
        printf("Boid went out of bounds, moved to x %lf.\n", newPosX);
    }
    if (newPosY + (image->height / 2) < 0) {
        newPosY = image->height + newPosY;
        printf("Boid went out of bounds, moved to y %lf.\n", newPosY);
    }
    else if (newPosY > (image->height / 2)) {
        newPosY = newPosY - image->height;
        printf("Boid went out of bounds, moved to y %lf.\n", newPosY);
    }

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
        int oldGridX = gridCoord[2*i];
        int oldGridY = gridCoord[2*i+1];

        if (oldGridX == gridX && oldGridY == gridY)
            continue;

        gridCoord[2*i] = gridX;
        gridCoord[2*i+1] = gridY;

        // Remove from the old bin data structure
        bin_t *oldBin = &(grid->bins[oldGridX * grid->cols + oldGridY]);
        entry_t *entry = oldBin->head;
        entry_t *my_entry = NULL;

        while (entry != NULL) {
            if (entry->boid == &(boids[i])) {
                my_entry = entry;
                
                if (my_entry->prev != NULL)
                    my_entry->prev->next = my_entry->next;
                
                if (my_entry->next != NULL)
                    my_entry->next->prev = my_entry->prev;
                
                if (my_entry == oldBin->head)
                    oldBin->head = my_entry->next;
                
                if (my_entry == oldBin->tail)
                    oldBin->tail = my_entry->prev;

                break;
            }
            entry = entry->next;
        }
        // ASSERT that my_entry will be non null after this
        if (my_entry == NULL) {
            printf("WARNING: my_entry is null.\n");
            my_entry = (entry_t *)malloc(sizeof(entry_t));
            my_entry->boid = &(boids[i]);
        }

        // Add to the new bin data structure
        bin_t *newBin = &(grid->bins[gridX * grid->cols + gridY]);
        if (newBin->tail) {
            // Add to the tail of the bin's current doubly linked list
            my_entry->prev = newBin->tail;
            my_entry->next = NULL;
            newBin->tail->next = my_entry;
            newBin->tail = my_entry;
        }
        else {
            // Add as first entry in bin (both head & tail)
            my_entry->prev = NULL;
            my_entry->next = NULL;
            newBin->head = my_entry;
            newBin->tail = my_entry;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

/* Update function */
void SeqBoids::updateScene() {
    // Parameters: need to make these modifiable outside of code
    float deltaT = 0.5f;
    float e = 75.0f;
    float s = -1.0f;
    float k = 0.8f;
    float m = 0.8f;

    // For sequential, we just iterate over all boids in order
    int boidCount = boid_group->count;
    for (int i = 0; i < boidCount; i++) {
        update_boid(i, deltaT, e, s, k, m, image);
    }

    // After that's finished, update the data structure
    update_grid(image);
}

/* Input function */
void SeqBoids::setup(const char *inputFile) {
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
    }

    // TODO: Establish parameters/test how the grid should be split
   
    // Initialize the spatial partitioning data structure
    grid = (grid_t*)malloc(sizeof(grid_t));
    grid->cellWidth = 10;
    grid->cellHeight = 10;
    grid->rows = image->height / grid->cellHeight;
    grid->cols = image->width / grid->cellWidth;
    // ASSERT: width and height are evenly divided into cells
    grid->bins = (bin_t*)calloc(grid->rows * grid->cols, sizeof(bin_t));
   
    // Assign each boid to its initial bin
    // For now, have an array of grid ID that a boid can check,
    // since it's not currently represented in the boid data struct.
    gridCoord = (int *)calloc(2 * num_of_boids, sizeof(int));
 
    for (int i = 0; i < num_of_boids; i++) {
        // Find which grid cell bin this boid should be in
        int gridX = (boids[i].position.x + (image->width / 2)) / grid->cellWidth;
        int gridY = (boids[i].position.y + (image->height / 2)) / grid->cellHeight;
        gridCoord[2*i] = gridX;
        gridCoord[2*i+1] = gridY;

        // Add to the bin data structure
        bin_t *bin = &(grid->bins[gridX * grid->cols + gridY]);
        
        entry_t *my_entry = (entry_t*)malloc(sizeof(entry_t));
        my_entry->boid = &(boids[i]);

        if (bin->tail) {
            // Add to the tail of the bin's current doubly linked list
            my_entry->prev = bin->tail;
            my_entry->next = NULL;
            bin->tail->next = my_entry;
            bin->tail = my_entry;
        }
        else {
            // Add as first entry in bin (both head & tail)
            my_entry->prev = NULL;
            my_entry->next = NULL;
            bin->head = my_entry;
            bin->tail = my_entry;
        }
    }
}

/* Output function */
Image *SeqBoids::output() {
    // Already the data that we're operating on
    return image;
}
