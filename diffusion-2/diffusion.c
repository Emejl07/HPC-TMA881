#include <stdio.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

int n_iter;
float diffusion_const;

//TODO: take in data file, parse it and assign to an initial a value
//TODO: width vs height sweep on bigger data - Locality
//TODO: Output average temperature - Reduction
//TODO: Output the average absolute difference of each temperature to the average of all temperatures

int main(int argc, char* argv[]) {
    // Parsing command line arguments
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

    // Initialize OpenCL
    cl_int error;

    cl_platform_id platform_id;
    cl_uint nmb_platforms;
    if (clGetPlatformIDs(1, &platform_id, &nmb_platforms) != CL_SUCCESS) {
        fprintf(stderr, "cannot get platform\n");
        return 1;
    }

    cl_device_id device_id;
    cl_uint nmb_devices;
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nmb_devices) != CL_SUCCESS) {
        fprintf(stderr, "cannot get device\n");
        return 1;
    }

    cl_context context;
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create context\n");
        return 1;
    }

    cl_command_queue command_queue;
    command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create command queue\n");
        return 1;
    }

    // Load kernel source code from file
    char *opencl_program_src;
    {
        FILE *clfp = fopen("./diffusion.cl", "r");
        if (clfp == NULL) {
            fprintf(stderr, "could not load cl source code\n");
            return 1;
        }
        fseek(clfp, 0, SEEK_END);
        int clfsz = ftell(clfp);
        fseek(clfp, 0, SEEK_SET);
        opencl_program_src = (char*)malloc((clfsz + 1) * sizeof(char));
        fread(opencl_program_src, sizeof(char), clfsz, clfp);
        opencl_program_src[clfsz] = 0;
        fclose(clfp);
    }

    // Initialize OpenCL program
    cl_program program;
    size_t src_len = strlen(opencl_program_src);
    program = clCreateProgramWithSource(context, 1, (const char **)&opencl_program_src, (const size_t *)&src_len, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create program\n");
        return 1;
    }

    free(opencl_program_src);

    // Build OpenCL program
    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot build program. log:\n");

        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *log = malloc(log_size * sizeof(char));
        if (log == NULL) {
            fprintf(stderr, "could not allocate memory\n");
            return 1;
        }

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        fprintf(stderr, "%s\n", log);

        free(log);

        return 1;
    }

    // Initialize OpenCL kernels
    cl_kernel kernel = clCreateKernel(program, "diffusion", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create kernel\n");
        return 1;
    }

    cl_kernel kernelBackwards = clCreateKernel(program, "diffusion", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create kernel\n");
        return 1;
    }

    cl_kernel kernel_sum = clCreateKernel(program, "reduction_sum", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create kernel reduction\n");
        return 1;
    }

    cl_kernel kernel_abs_diff = clCreateKernel(program, "absolute_difference", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create kernel reduction\n");
        return 1;
    }

    //--------------------------------------------------------------------------------
    // Parameters and buffers

    FILE *fp = fopen("init", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open the file.\n");
        return 1;
    }
    int width, height;

    fscanf(fp, "%d %d", &width, &height);
    width += 2;
    height += 2;

    float *M = malloc(width * height * sizeof(float));
    for (int ix = 0; ix < width * height; ++ix)
        M[ix] = 0;

    int x, y;
    double value;
    while (fscanf(fp, "%d %d %lf", &x, &y, &value) == 3) {
        M[(x + 1) * height + (y + 1)] = value;
    }

    cl_mem input_buffer, output_buffer, output_buffer_sum;
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create buffer a\n");
        return 1;
    }

    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create buffer c\n");
        return 1;
    }

    const int sz = width * height;
    const int global_redsz = 64;
    const int local_redsz = 8;
    const int nmb_redgps = global_redsz / local_redsz;

    output_buffer_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nmb_redgps * sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "cannot create buffer c_sum\n");
        return 1;
    }

    // Assign input values to the kernels
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(float), &diffusion_const);

    clSetKernelArg(kernelBackwards, 0, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernelBackwards, 1, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernelBackwards, 2, sizeof(int), &width);
    clSetKernelArg(kernelBackwards, 3, sizeof(float), &diffusion_const);

    // Writes the initial input into input buffer
    if (clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, width * height * sizeof(float), M, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue write of buffer a\n");
        return 1;
    }

    const size_t global_sz[] = { width - 2, height - 2 };
    const size_t local_sz[] = { 10,10 };

    int iter;
    for (iter = 0; iter + 1 < n_iter; iter += 2) {
        // Enqueue kernel (what should the GPU do)
        if (clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, (const size_t *)global_sz, (const size_t *)local_sz, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue kernel\n");
            return 1;
        }
        if (clEnqueueNDRangeKernel(command_queue, kernelBackwards, 2, NULL, (const size_t *)global_sz, (const size_t *)local_sz, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue kernel\n");
            return 1;
        }
    }
    if (iter < n_iter) {
        if (clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, (const size_t *)global_sz, NULL, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue kernel\n");
            return 1;
        }
    }

    // Enqueue buffer (what arrays should the GPU do the stuff on)
    if (n_iter % 2 == 0) {
        if (clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, width * height * sizeof(float), M, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue read of buffer c\n");
            return 1;
        }
    }
    else {
        if (clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, width * height * sizeof(float), M, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue read of buffer c\n");
            return 1;
        }
    }

    // Reduction stuff
    const cl_int sz_clint = (cl_int)sz;
    clSetKernelArg(kernel_sum, 0, sizeof(cl_mem), &input_buffer); //TODO generalize
    clSetKernelArg(kernel_sum, 1, local_redsz * sizeof(float), NULL);
    clSetKernelArg(kernel_sum, 2, sizeof(cl_int), &sz_clint);
    clSetKernelArg(kernel_sum, 3, sizeof(cl_mem), &output_buffer_sum);

    size_t global_redsz_szt = (size_t)global_redsz;
    size_t local_redsz_szt = (size_t)local_redsz;
    if (clEnqueueNDRangeKernel(command_queue, kernel_sum, 1, NULL, (const size_t *)&global_redsz_szt, (const size_t *)&local_redsz_szt, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue kernel reduction\n");
        return 1;
    }

    float *sum = malloc(nmb_redgps * sizeof(float));
    if (clEnqueueReadBuffer(command_queue, output_buffer_sum, CL_TRUE, 0, nmb_redgps * sizeof(float), sum, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue read of buffer c\n");
        return 1;
    }

    for (int i = 1; i < nmb_redgps; i++) {
        sum[0] += sum[i];
    }
    sum[0] /= ((float)width) * ((float)height);

    // Absolute difference
    clSetKernelArg(kernel_abs_diff, 0, sizeof(cl_mem), &input_buffer); //TODO generalize
    clSetKernelArg(kernel_abs_diff, 1, sizeof(int), &height);
    clSetKernelArg(kernel_abs_diff, 2, sizeof(float), sum);

    if (clEnqueueNDRangeKernel(command_queue, kernel_abs_diff, 2, NULL, (const size_t *)global_sz, NULL, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue kernel\n");
        return 1;
    }

    if (n_iter % 2 == 0) {
        if (clEnqueueReadBuffer(command_queue, input_buffer, CL_TRUE, 0, width * height * sizeof(float), M, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue read of buffer c\n");
            return 1;
        }
    }
    else {
        if (clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, width * height * sizeof(float), M, 0, NULL, NULL) != CL_SUCCESS) {
            fprintf(stderr, "cannot enqueue read of buffer c\n");
            return 1;
        }
    }

    // Reduction sum of absolute difference array
    if (clEnqueueNDRangeKernel(command_queue, kernel_sum, 1, NULL, (const size_t *)&global_redsz_szt, (const size_t *)&local_redsz_szt, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue kernel reduction\n");
        return 1;
    }

    float *sumAbsDiff = malloc(nmb_redgps * sizeof(float));
    if (clEnqueueReadBuffer(command_queue, output_buffer_sum, CL_TRUE, 0, nmb_redgps * sizeof(float), sumAbsDiff, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "cannot enqueue read of buffer c\n");
        return 1;
    }

    for (int i = 1; i < nmb_redgps; i++) {
        sumAbsDiff[0] += sumAbsDiff[i];
    }
    sumAbsDiff[0] /= ((float)width) * ((float)height);

    // Finish
    if (clFinish(command_queue) != CL_SUCCESS) {
        fprintf(stderr, "cannot finish queue\n");
        return 1;
    }

    // Print results
    printf("%.2f \n", sum[0]);
    printf("%.2f\n", sumAbsDiff[0]);

    // Free resources
    free(M);
    free(sum);
    free(sumAbsDiff);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(output_buffer_sum);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseKernel(kernelBackwards);
    clReleaseKernel(kernel_sum);
    clReleaseKernel(kernel_abs_diff);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
