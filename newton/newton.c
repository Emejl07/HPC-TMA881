#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <threads.h>
#include "color_encodings.h"

// Global variables for user input arguments
int n_threads, sz, degree;
const long upper_bnd = 10000000000; // Upper bound constant
const float lower_bnd_squared = 0.000001f; // Lower bound squared constant
const int color_string_length = 12; // Length of color string "xxx xxx xxx "

// Explicit root solutions for each polynomial degree between 1 and 9
const float root_solutions[9][9][2] = {
  // Solutions for degree 1
  {{1.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 2
  {{1.f,0.f}, {-1.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 3
  {{1.f,0.f}, {-0.5f,0.86603f}, {-0.5f,-0.86603f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 4
  {{1.f,0.f}, {-1.f,0.f}, {0.f,1.f}, {0.f,-1.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 5
  {{1.f,0.f}, {0.30902f,0.95106f}, {0.30902f,-0.95106f}, {-0.80902f,0.58779f}, {-0.80902f,-0.58779f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 6
  {{1.f,0.f}, {-1.f,0.f}, {0.5f,0.86603f}, {-0.5f,-0.86603f}, {-0.5f,0.86603f}, {0.5f,-0.86603f}, {0.f,0.f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 7
  {{1.f,0.f}, {-0.90097f,-0.43388f}, {-0.90097f,0.43388f}, {-0.22252f,-0.97493f}, {-0.22252f,0.97493f}, {0.62349f,-0.78183f}, {0.62349f,0.78183f}, {0.f,0.f}, {0.f,0.f}},
  // Solutions for degree 8
  {{1.f,0.f}, {-1.f,0.f}, {0.f,1.f}, {0.f,-1.f}, {0.70711f,0.70711f}, {-0.70711f,-0.70711f}, {-0.70711f,0.70711f}, {0.70711f,-0.70711f}, {0.f,0.f} },
  // Solutions for degree 9
  {{1.f,0.f}, {-0.93969f,-0.34202f}, {-0.93969f,0.34202f}, {0.76604f,0.64279f}, {0.76604f,-0.64279f}, {-0.5f,-0.86603f}, {-0.5f,0.86603f}, {0.17365f,0.98481f}, {0.17365f,-0.98481f}}
};

// Structure for complex numbers
typedef struct{
  float re;
  float im;
} complex;

// Function to perform Newton step for a given polynomial degree
complex newton_step(int degree, float x, float y){
  float denom, x2y2, xnumer, ynumer;
  x2y2=x*x+y*y;
  switch(degree){
    case 1:
      return (complex){1.f, 0.f};
    case 2:
      denom = 2*x2y2;
      return (complex){x*0.5f + x/denom, y*0.5f - y/denom};
    case 3:
      denom = 3.f*x2y2*x2y2; 
      xnumer = x*x - y*y;
      ynumer = 2.f*x*y;
      return (complex){x*(2.f/3.f) + (xnumer)/denom,y*(2.f/3.f) - ynumer/denom};
    // Cases for other degrees omitted for brevity...
    default:
      fprintf(stderr, "unexpected degree\n");
      exit(1);
  }
}

// Structure for returning root and iteration count
typedef struct {
  uint8_t root;
  uint8_t iter;
} return_tuple;

// Function to run the Newton algorithm for a given starting position in the complex plane
return_tuple newton_algorithm(float re, float im, int degree)
{
  float dre, dim;

  for (uint8_t i = 0;; ++i) {
    // Check if the iteration count exceeds the limit or the point diverges
    if(fabs(re)>upper_bnd || fabs(im)>upper_bnd || re*re+im*im<lower_bnd_squared || i==128){
      return (return_tuple){10, i};
    }
    // Check for convergence to each root
    for (int8_t j = 0; j < degree; j++)
    {
      dre=re-root_solutions[degree-1][j][0];
      dim=im-root_solutions[degree-1][j][1];
      if (dre * dre + dim * dim < 0.000001f)
      {
        return (return_tuple){j+1, i};
      }
    }

    // Perform Newton step
    complex post_step = newton_step(degree, re, im);
    re = post_step.re;
    im = post_step.im;
  }
}

// Structure for passing information to computation threads
typedef struct {
  const float **re;
  const float **im;
  uint8_t **convergences;
  uint8_t **attractors;
  int ib;
  int istep;
  int sz;
  int tx;
  mtx_t *mtx;
  cnd_t *cnd;
  int_padded *status;
} thrd_info_t;

// Structure for passing information to the check thread
typedef struct {
  const float **re;
  const float **im;
  uint8_t **convergences;
  uint8_t **attractors;
  int sz;
  int n_threads;
  mtx_t *mtx;
  cnd_t *cnd;
  int_padded *status;
} thrd_info_check_t;

// Function to be executed by computation threads
int main_thrd(void *args)
{
  const thrd_info_t *thrd_info = (thrd_info_t*) args;
  const float **im = thrd_info->im;
  const float **re = thrd_info->re;
  uint8_t **convergences = thrd_info->convergences;
  uint8_t **attractors = thrd_info->attractors;
  const int ib = thrd_info->ib;
  const int istep = thrd_info->istep;
  const int sz = thrd_info->sz;
  const int tx = thrd_info->tx;
  mtx_t *mtx = thrd_info->mtx;
  cnd_t *cnd = thrd_info->cnd;
  int_padded *status = thrd_info->status;

  // Loop over all rows the thread will compute
  for (int ix = ib; ix < sz; ix += istep) {
    const float *reix = re[ix];
    const float *imix = im[ix];
    // Allocate memory for the rows of the result before computing
    uint8_t *attractor = (uint8_t*) malloc(sz*sizeof(uint8_t));
    uint8_t *convergence = (uint8_t*) malloc(sz*sizeof(uint8_t));

    // Loop over each element in the row
    for (int jx = 0; jx < sz; ++jx) {
      // Perform Newton algorithm for each element
      return_tuple values = newton_algorithm(reix[jx],imix[jx],degree);
      attractor[jx] = values.root;
      convergence[jx] = values.iter;
    }

    // Lock the mutex before updating shared data
    mtx_lock(mtx); 
    convergences[ix] = convergence;
    attractors[ix] = attractor;
    status[tx].val = ix + istep;
    // Unlock the mutex after updating shared data
    mtx_unlock(mtx);
    // Signal the checker that a line is finished and can be checked
    cnd_signal(cnd);
  }

  return 0;
}

// Function to be executed by the check thread
int main_thrd_write(void *args)
{
  const thrd_info_check_t *thrd_info = (thrd_info_check_t*) args;
  const float **im = thrd_info->im;
  const float **re = thrd_info->re;
  uint8_t **convergences = thrd_info->convergences;
  uint8_t **attractors = thrd_info->attractors;
  const int sz = thrd_info->sz;
  const int n_threads = thrd_info->n_threads;
  mtx_t *mtx = thrd_info->mtx;
  cnd_t *cnd = thrd_info->cnd;
  int_padded *status = thrd_info->status;
  
  int color_max=-1;
  char attractor_line_string[sz*color_string_length];
  char convergence_line_string[sz*color_string_length];

  // Open files for writing attractors and convergence data
  char name_file[26];
  snprintf(name_file, sizeof(name_file), "newton_attractors_x%d.ppm", degree);
  FILE* fp = fopen(name_file, "w");
  fprintf(fp, "P3\n%i %i\n255\n", sz, sz);

  snprintf(name_file, sizeof(name_file), "newton_convergence_x%d.ppm", degree);
  FILE* fp2 = fopen(name_file, "w");
  fprintf(fp2, "P3\n%i %i\n255\n", sz, sz);

  // Loop until all lines are processed
  for (int ix = 0, ibnd; ix < sz; ) {
    // Wait until new lines are available
    for (mtx_lock(mtx); ; ) {
      ibnd = sz;
      // Find the minimum of all status variables
      for (int tx = 0; tx < n_threads; ++tx)
        if (ibnd > status[tx].val)
          ibnd = status[tx].val;

      if (ibnd <= ix)
        cnd_wait(cnd,mtx);
      else {
        mtx_unlock(mtx);
        break;
      }
    }

    // Loop through the lines and process them
    for (; ix < ibnd; ++ix) {
      // Find the maximum color value for normalization
      for (int j = 0; j < sz; j++){
        if(convergences[ix][j]>color_max && ix == 0)
          color_max = convergences[ix][j];
      }

      // Convert attractor and convergence data to color strings
      for (int j = 0; j < sz; j++){
        memcpy(attractor_line_string + j*color_string_length, root_encoding[attractors[ix][j]], color_string_length);
        memcpy(convergence_line_string + j*color_string_length, convergence_colors[(128/color_max)*convergences[ix][j]], color_string_length);
      }
      // Write attractor and convergence data to files
      fwrite(attractor_line_string, 1, sz*color_string_length,fp);
      fwrite(convergence_line_string, 1, sz*color_string_length,fp2);

      // Free memory allocated for attractor and convergence data
      free(attractors[ix]);
      free(convergences[ix]);
    } 
  }
  // Close files after writing
  fclose(fp);
  return 0;
}

int main(int argc, char*argv[])
{
    // Parsing command line arguments
    int opt;
    while((opt = getopt(argc, argv, "t: l:")) != -1){
        switch(opt){
            case 't':
                n_threads = atoi(optarg);
                break;
            case 'l':
                sz = atoi(optarg);
                break;
            default:
                break;
        }
    }
    if (argv[3] != NULL){
        degree = atoi(argv[argc-1]);
    }
    else{
        printf("No exponent degree was given \n");
        return 0;
    }

    // Check if the number of threads exceeds the number of rows
    if(n_threads>sz){
      printf("You can't have more threads than the number of rows in the picture");
      return 0;
    }
    printf("Number of threads:%i, Number of rows and cols: %i, exponent of x^: %i \n", n_threads, sz, degree);

    // Allocate memory for arrays
    float **re = (float**) malloc(sz*sizeof(float*));
    float **im = (float**) malloc(sz*sizeof(float*));
        uint8_t **attractors = (uint8_t**) malloc(sz*sizeof(uint8_t*));
    uint8_t **convergences = (uint8_t**) malloc(sz*sizeof(uint8_t*));
    float *reentries = (float*) malloc(sz*sz*sizeof(float));
    float *imentries = (float*) malloc(sz*sz*sizeof(float));

    // Assign memory for the 2D arrays
    for (int ix = 0, jx = 0; ix < sz; ++ix, jx += sz) {
        re[ix] = reentries + jx;
        im[ix] = imentries + jx;
    }

    // Initialize the complex plane
    for (int ire = 0; ire < sz; ire++) {
        for (int iim = 0; iim < sz; iim++) {
            re[iim][ire] = ((float)ire) / ((float) sz) * 4.f - 2.f;
            im[iim][ire] = ((float)iim) / ((float) sz) * 4.f - 2.f;
        }
    }

    thrd_t thrds[n_threads];
    thrd_info_t thrds_info[n_threads];

    thrd_t thrd_check;
    thrd_info_check_t thrd_info_check;

    mtx_t mtx;
    mtx_init(&mtx, mtx_plain);

    cnd_t cnd;
    cnd_init(&cnd);

    int_padded status[n_threads];

    // Compute attractors and convergence using multiple threads
    for (int tx = 0; tx < n_threads; ++tx) {
        thrds_info[tx].im = (const float**) im;
        thrds_info[tx].re = (const float**) re;
        thrds_info[tx].convergences = convergences;
        thrds_info[tx].attractors = attractors;
        thrds_info[tx].ib = tx;
        thrds_info[tx].istep = n_threads; // Each thread processes its own set of rows
        thrds_info[tx].sz = sz;
        thrds_info[tx].tx = tx;
        thrds_info[tx].mtx = &mtx;
        thrds_info[tx].cnd = &cnd;
        thrds_info[tx].status = status;
        status[tx].val = -1;

        int r = thrd_create(thrds + tx, main_thrd, (void*) (thrds_info + tx));
        if (r != thrd_success) {
            fprintf(stderr, "failed to create thread\n");
            exit(1);
        }
        thrd_detach(thrds[tx]); // Detach each thread once it's done
    }

    // Launch a thread to check computation and free allocated memory
    {
        thrd_info_check.im = (const float**) im;
        thrd_info_check.re = (const float**) re;
        thrd_info_check.convergences = convergences;
        thrd_info_check.attractors = attractors;
        thrd_info_check.sz = sz;
        thrd_info_check.n_threads = n_threads;
        thrd_info_check.mtx = &mtx;
        thrd_info_check.cnd = &cnd;
        // Initialize status in the previous loop since it will be used by the check thread
        thrd_info_check.status = status;

        int r = thrd_create(&thrd_check, main_thrd_write, (void*) (&thrd_info_check));
        if (r != thrd_success) {
            fprintf(stderr, "failed to create thread\n");
            exit(1);
        }
    }

    // Wait for the check thread to finish, indicating that all threads are done
    {
        int r;
        thrd_join(thrd_check, &r);
    }

    // Free allocated memory
    free(reentries);
    free(imentries);
    free(re);
    free(im);
    free(attractors);
    free(convergences);

    // Destroy mutex and conditional variable
    mtx_destroy(&mtx);
    cnd_destroy(&cnd);

    return 0;
}
