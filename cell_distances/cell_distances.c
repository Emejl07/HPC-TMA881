#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

// Function prototypes
static inline void parse_coord(int16_t*, char*);
void parse_points(int16_t*, char*, int);
static inline int distances_3d(int16_t*, int16_t*);

// Constants
const int row_size = 24;
const int cols = 3;
const int max_dist = 3465; // there are 20*sqrt(3) possible values

// Function to parse a single coordinate from a string
static inline 
void parse_coord(int16_t* coord, char* const str)
{
    // Determine if the coordinate is negative
    if (str[0] == 45){
        *coord = -((str[1] - '0') * (int16_t)10000 +
                   (str[2] - '0') * (int16_t)1000 +
                   (str[4] - '0') * (int16_t)100 +
                   (str[5] - '0') * (int16_t)10 +
                   (str[6] - '0'));
    }
    else{
        *coord = ((str[1] - '0') * (int16_t)10000 +
                  (str[2] - '0') * (int16_t)1000 +
                  (str[4] - '0') * (int16_t)100 +
                  (str[5] - '0') * (int16_t)10 +
                  (str[6] - '0'));
    }
}

// Function to parse a string containing multiple points
void parse_points(int16_t* arr, char* const str, int n_rows)
{
    int16_t *coord;
    char *coord_str;
    for (coord = arr, coord_str = str;
         coord != arr + 3 * n_rows; // 3*nmbpts;
         ++coord, coord_str += 8)
        parse_coord(coord, coord_str);
}

// Function to calculate the distance between two points in 3D space
static inline 
int distances_3d(int16_t* vec1, int16_t* vec2)
{
    float dx = (float)(vec1[0] - vec2[0]);
    float dy = (float)(vec1[1] - vec2[1]);
    float dz = (float)(vec1[2] - vec2[2]);
    
    // Calculates distance and converts back to int (truncating to 2 decimal places)
    return (int)(sqrtf(dx * dx + dy * dy + dz * dz) * 0.1f);
}

int main(int argc, char* argv[]){
    // Determine the number of threads from input arg
    int opt, n_threads;
    while((opt = getopt(argc, argv, "t:")) != -1){
        switch(opt){
            case 't':
                n_threads = atoi(optarg);
                break;
            default:
                break;
        }
    }
    omp_set_num_threads(n_threads);

    // Open the file
    FILE *fp = fopen("cells", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Determine size of the file and extract row size and block size from that
    fseek(fp, 0L, SEEK_END);
    uint32_t rows = ftell(fp)/row_size;
    fclose(fp);
    const int block_size = (int) fmin(rows * 0.01, 100000.f); // size of one block
    const int max_read_size = block_size * 2; // maximum amount of located memory
    const int iter = (rows - 1) / block_size + 1;
    const int last_block_size = rows - block_size * (iter - 1);

    // Declare arrays
    int16_t* asentries = (int16_t*)malloc(sizeof(int16_t) * max_read_size * cols);
    int16_t** cells = (int16_t**)malloc(sizeof(int16_t*) * max_read_size);
    for (size_t i = 0, j = 0; i < max_read_size; i++, j += cols)
        cells[i] = asentries + j;

    size_t distances[max_dist]; 
    for (int16_t i = 0; i < max_dist; i++){
        distances[i] = 0;
    }
    fp = fopen("cells", "r");
    char str_to_parse[block_size * row_size];
    int bz, mxrdsz;
    for (int k = 0; k < iter; k++){   
        bz = k != iter - 1 ? block_size : last_block_size;

        fseek(fp, bz * row_size * k, SEEK_SET); // Finds starting point of reading
        fread((void*) str_to_parse, sizeof(char), bz * row_size, fp); // Reads a certain amount of rows
        parse_points(cells[0], str_to_parse, bz);

        for (int i = 0; i < bz - 1; i++){
            for (int j = i + 1; j < bz; j++){
                // Calculate distances and update the distances array
                distances[distances_3d(cells[i], cells[j])]++;
            }
        }
        
        // All the cross read ins and distance calculations
        for (int ic = k + 1; ic < iter; ic++){
            if (ic != iter - 1){
                bz = block_size;
                mxrdsz = max_read_size;
            }
            else{
                bz = last_block_size;
                mxrdsz = block_size + last_block_size;
            }

            fseek(fp, bz * row_size * ic, SEEK_SET); // Finds starting point of reading
            fread((void*) str_to_parse, sizeof(char), bz * row_size, fp); // Reads a certain amount of rows
            parse_points(cells[block_size], str_to_parse, bz);

            #pragma omp parallel for shared(cells) reduction(+:distances[:max_dist])
            for (int iown = 0; iown < bz; iown++){
                for (int icross = block_size; icross < mxrdsz; icross++){
                    // Calculate distances and update the distances array
                    distances[distances_3d(cells[iown], cells[icross])]++;
                }
            }
        }
    }
    fclose(fp);

    // Print out all the distances and corresponding frequencies (excluding all duplicates)
    for (int16_t i = 0; i < max_dist; i++){
        if (distances[i] != 0){
            if (i < 1000)
                printf("0%.2f %d \n", i * 0.01f, distances[i]);
            else
                printf("%.2f %d \n", i * 0.01f, distances[i]);
        }
    }
    
    free(asentries);
    free(cells);
    return 0;
}
