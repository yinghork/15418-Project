#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "seqBoids.h"
#include "openmpBoids.h"
#include "cudaBoids.h"

static int _argc;
static const char **_argv;

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
    printf("\t-r <seq/cuda>\n");
    printf("\t-n <num_of_threads>\n");
    printf("\t-i <SA_iters>\n");
}

void writeOutput(Image *image, int iter){
    char filename[1024];
    sprintf(filename, "./output/%s_%d.txt", "framePositions", iter);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    fprintf(fp, "%d %d\n", image->width, image->height);

    int num_boids = image->data->count;
    boid_t *boids = image->data->boids;

    for (int i = 0; i < num_boids; i++) {
        fprintf(fp, "%lf %lf", boids[i].position.x, boids[i].position.y);
        if (i != num_boids - 1)
           fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Wrote boids frame file %s\n", filename);
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *algorithm_name = get_option_string("-r", "seq");
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
    printf("Algorithm in use: %s\n", algorithm_name);

    /* Run the boids algorithm */

    Boids *boidsAlgorithm;
    if (strcmp(algorithm_name, "openmp") == 0){
        boidsAlgorithm = new OpenmpBoids();
    }
    else if (strcmp(algorithm_name, "cuda") == 0) {
        boidsAlgorithm = new CudaBoids();
    }
    else
        boidsAlgorithm = new SeqBoids();

    // Load the initial state of the scene from the input file
    boidsAlgorithm->setup(input_filename, num_of_threads);

    // Output the first frame of animation (initial state)
    Image *image = boidsAlgorithm->output();
    writeOutput(image, 0);

    double compute_time = 0;

    // Pass over all boids in each iteration, computing next time step 
    for (int iter = 1; iter < SA_iters; iter++) {
        auto compute_start = Clock::now();
        
        boidsAlgorithm->updateScene();

        compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
        
        // Output the next frame of the animation
        image = boidsAlgorithm->output();
        writeOutput(image, iter);
    }

    printf("Total computation Time (ms): %lf.\n", compute_time * 1000);
    printf("Per iter computation Time (ms): %lf.\n", compute_time/SA_iters * 1000);

    return 0;
}
