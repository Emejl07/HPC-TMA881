#define _XOPEN_SOURCE 700

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

// Declare global variables
int n_iter, width, height, sz;
float diffusion_const;
int scatter_root = 0;

int main(int argc, char * argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, nmb_mpi_proc;
    // Get MPI rank and number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nmb_mpi_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Define the total number of elements

    // Create an array on the root process (mpi_rank 0)
    float *M;
    if (mpi_rank == scatter_root) {
        // Parse command line arguments and read data from file
        int opt;
        while ((opt = getopt(argc, argv, "n: d:")) != -1) {
            switch (opt) {
                case 'n':
                    n_iter = atoi(optarg);
                    break;
                case 'd':
                    diffusion_const = atof(optarg);
                    break;
                default:
                    break;
            }
        }

        FILE *fp = fopen("init", "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: Could not open the file.\n");
            return 1;
        }

        fscanf(fp, "%d %d", &width, &height);
        width += 2;
        height += 2;

        sz = height * width;

        M = malloc(sz * sizeof(float));
        for (int ix = 0; ix < sz; ++ix)
            M[ix] = 0;

        int x, y;
        float value;
        while (fscanf(fp, "%d %d %f", &x, &y, &value) == 3) {
            M[(x + 1) * height + (y + 1)] = value;
        }
    }
    // Broadcast necessary data to all processes
    MPI_Bcast(&sz, 1, MPI_INT, scatter_root, MPI_COMM_WORLD);
    MPI_Bcast(&diffusion_const, 1, MPI_FLOAT, scatter_root, MPI_COMM_WORLD);
    MPI_Bcast(&n_iter, 1, MPI_INT, scatter_root, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, scatter_root, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, scatter_root, MPI_COMM_WORLD);
    
    // Calculate local height for each process
    int height_loc = (height-1) / nmb_mpi_proc + 1;

    // Calculate positions and lengths for Scatterv and Gatherv
    int pos, poss[nmb_mpi_proc];
    int len, lens[nmb_mpi_proc];

    for (int jx = 0, pos = -width; jx < nmb_mpi_proc; ++jx, pos += height_loc*width) {
        poss[jx] = pos;
        lens[jx] = (height_loc+2)*width;
    }
    poss[0] = 0;
    lens[0] = (height_loc+1)*width;
    lens[nmb_mpi_proc-1] = sz - poss[nmb_mpi_proc-1];

    height_loc = lens[mpi_rank] / width;
    
    // Create arrays to store local elements on each process
    float *M_loc_in = (float *)malloc(height_loc*width * sizeof(float));
    float *M_loc_out = (float *)malloc(height_loc*width * sizeof(float));

    // Initialize local arrays with zeros
    for (int i = 0; i < height_loc*width; i++) {
        M_loc_in[i] = 0.;
        M_loc_out[i] = 0.;
    }

    // Scatter data from root process to all processes
    MPI_Scatterv(M, lens, poss, MPI_FLOAT, M_loc_in, (height_loc+2)*width, MPI_FLOAT,
                 scatter_root, MPI_COMM_WORLD);

    int iter;
    float *M_loc_temp;
    for (iter = 0; iter < n_iter; iter++) {
        // Algorithm step
        for (int i = 1; i < height_loc-1; i++) {
            for (int j = 1; j < width-1; j++) {
                M_loc_out[j+i*width] = M_loc_in[j+i*width] + diffusion_const * ((M_loc_in[j-1+i*width] + M_loc_in[j+1+i*width] + M_loc_in[j+(i+1)*width] + M_loc_in[j+(i-1)*width]) * 0.25f - M_loc_in[j+i*width]);
            }
        }
        // Send last row down
        if (mpi_rank != nmb_mpi_proc - 1) { 
            MPI_Sendrecv(M_loc_out + (height_loc-2)*width, width, MPI_FLOAT, mpi_rank + 1, 0, M_loc_in + (height_loc-1)*width, width, MPI_FLOAT, mpi_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Send first row up
        if (mpi_rank != 0) {
            MPI_Sendrecv(M_loc_out+width, width, MPI_FLOAT, mpi_rank - 1, 0, M_loc_in, width, MPI_FLOAT, mpi_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Swap pointers for M_loc_in and M_loc_out
        M_loc_temp = M_loc_in;
        M_loc_in = M_loc_out;
        M_loc_out = M_loc_temp;
    }

    // Calculate partial sum for each process
    float sum = 0.f;
    float partial_sum = 0.f;
    for (int i = width; i < (height_loc-1)*width; i++) {
        partial_sum += M_loc_in[i];
    }
    partial_sum /= ((float)(width-2)) * ((float)(height_loc-2));

    // Gather partial sums to the root process
    MPI_Reduce(&partial_sum, &sum, 1, MPI_FLOAT, MPI_SUM, scatter_root, MPI_COMM_WORLD);
    sum /= nmb_mpi_proc;

    // Calculate absolute differences
    MPI_Bcast(&sum, 1, MPI_FLOAT, scatter_root, MPI_COMM_WORLD);

    float absdiff_sum = 0.f;
    float absdiff_partial_sum = 0.f;
    for (int i = width; i < (height_loc-1)*width; i++) {
        if (M_loc_in[i] > sum)
            absdiff_partial_sum += (M_loc_in[i]-sum);
        else
            absdiff_partial_sum += (sum-M_loc_in[i]);
    }
    absdiff_partial_sum /= ((float)(width-2)) * ((float)(height_loc-2));

    // Gather absolute difference partial sums to the root process
    MPI_Reduce(&absdiff_partial_sum, &absdiff_sum, 1, MPI_FLOAT, MPI_SUM, scatter_root, MPI_COMM_WORLD);
    absdiff_sum /= nmb_mpi_proc;
    
    // Print results on root process
    if (mpi_rank == scatter_root) {
        printf("%.2f\n", sum);
        printf("%.2f\n", absdiff_sum);
    }

    // Clean up memory and finalize MPI
    free(M_loc_in);
    free(M_loc_out);
    if (mpi_rank == scatter_root) {
        free(M);
    }
    MPI_Finalize();
    return 0;
}

