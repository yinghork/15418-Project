

#include "wireroute.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

static int _argc;
static const char **_argv;

group_t* boid_group;


const char *get_option_string(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-i <SA_iters>\n");
}

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
    int distance = 100; // Threshold of distance between boids
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
    
    // 0.5% towards the percieved center
    vel_t new_vel;
    new_vel.x = (result.x - boids[own_i].velocity.x) / 10;
    new_vel.y = (result.y - boids[own_i].velocity.y) / 10;
    return new_vel;

}

static void update_boid_pos(group_t* group, int own_i) {

    boid_t* boids = group->boids;

    pos_t p1 = flock_center_rule(group, own_i);
    pos_t p2 = collision_avoidance_rule(group, own_i);
    vel_t v3 = velocity_matching_rule(group, own_i);

    boids[own_i].velocity.x += p1.x + p2.x + v3.x; 
    boids[own_i].velocity.y += p1.y + p2.y + v3.y;

    // Update the position to the new position
    boids[own_i].position.x += boids[own_i].velocity.x; 
    boids[own_i].position.y += boids[own_i].velocity.y; 
}

static void draw_boid(){
}


int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    int SA_iters = get_option_int("-i", 5);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Number of simulated annealing iterations: %d\n", SA_iters);
    printf("Input file: %s\n", input_filename);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    // Open output file
    char outputFileName[120];
    int err = snprintf(outputFileName, sizeof(char)*100, "costs_%s_%d.txt", input_filename, num_of_threads);
    if (err < 0) {
        printf("Unable to create output file.\n");
        return 1;
    }
    FILE *outputFile = fopen(outputFileName, "w");
    if (!outputFile) {
        printf("Unable to create file: %s.\n", outputFileName);
        return 1;
    }

    fprintf(outputFile, "%d %d\n", dim_x, dim_y);
    fprintf(outputFile, "%d\n", num_of_boids);
    for (int i = 0; i < dim_x; i++) {
        for (int j = 0; j < dim_y; j++) {
            fprintf(costFile, "%d ", costs[i*dim_y + j]);
        }
        if (i != dim_x - 1)
           fprintf(costFile, "\n");
    }


    int dim_x, dim_y;
    int num_of_boids;

    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    fscanf(input, "%d\n", &num_of_boids);

    // Allocate mem for the boids
    group_t* group = (group_t*)malloc(sizeof(group_t));
    boid_t *boids = (boid_t *)calloc(num_of_boids, sizeof(boid_t));
    group->boids = boids;
    group->size = num_of_boids;
    
    /* Read the grid dimension and boid information from file */
    
    // Load the coords (x1,y1),(x2,y2) for each boid
    int x1, y1;
    for (int i = 0; i < num_of_boids; i++) {
        fscanf(input, "%d %d\n", &x1, &y1);
        boids[i].position.x = x1;
        boids[i].position.y = y1;
    }
    

    // Set the number of threads used by openmp
    omp_set_num_threads(num_of_threads);

    auto compute_start = Clock::now();
    double compute_time = 0;

    // Pass over all boids in each iteration, computing next time step 
    for (int iter = 0; iter < SA_iters; iter++) {

        int i;
        // #pragma omp parallel for default(shared) private(i) schedule(dynamic) 
        for (i = 0; i < num_of_boids; i++) {
            drawBoid();
            update_boid_pos(group, i);
        }
    }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    /* Write output to files */

    return 0;
}
