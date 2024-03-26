__kernel
void
diffusion(
    __global const float *h_in, // Input buffer
    __global float *h_out,      // Output buffer
    //int h,                    // Height
    int w,                      // Width
    float diff_const            // Diffusion constant
    )
{
    // Calculate global indices
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    // Apply diffusion formula
    h_out[i + j * w] = h_in[i + j * w] + diff_const * (
        (h_in[(i - 1) + j * w] + h_in[(i + 1) + j * w] +
        h_in[i + (j + 1) * w] + h_in[i + (j - 1) * w]) * 0.25f - h_in[i + j * w]);
}

__kernel
void
reduction_sum(
    __global float *M_in,       // Input buffer
    __local float *scratch,     // Local scratch buffer
    __const int sz,             // Size of input buffer
    __global float *sum         // Output sum
    )
{
    // Get global and local sizes and indices
    int gsz = get_global_size(0);
    int gix = get_global_id(0);
    int lsz = get_local_size(0);
    int lix = get_local_id(0);

    // Compute partial sum
    float acc = 0;
    for (int cix = get_global_id(0); cix < sz; cix += gsz)
        acc += M_in[cix];

    // Store partial sum in local memory
    scratch[lix] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce partial sums in local memory
    for (int offset = lsz / 2; offset > 0; offset /= 2) {
        if (lix < offset)
            scratch[lix] += scratch[lix + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final sum to global memory
    if (lix == 0)
        sum[get_group_id(0)] = scratch[0];
}

__kernel
void
absolute_difference(
    __global float *M,  // Input/output buffer
    int h,              // Height
    float sum           // Sum value
    )
{
    // Calculate global indices
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    // Compute absolute difference
    if (M[i * h + j] > sum)
        M[i * h + j] = M[i * h + j] - sum;
    else
        M[i * h + j] = sum - M[i * h + j];
}
