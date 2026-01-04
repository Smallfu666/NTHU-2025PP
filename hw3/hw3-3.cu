#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define B 64
#define TB 32
const int INF = 1073741823;

int n, m, padded_n;
int *Dist = NULL;

__global__ void phase1(int *dist, int n, int Round)
{
    __shared__ int shared[B][B];
    int base_idx = Round * B * n + Round * B;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    shared[ty][tx] = dist[base_idx + ty * n + tx];
    shared[ty][tx + TB] = dist[base_idx + ty * n + (tx + TB)];
    shared[ty + TB][tx] = dist[base_idx + (ty + TB) * n + tx];
    shared[ty + TB][tx + TB] = dist[base_idx + (ty + TB) * n + (tx + TB)];
    __syncthreads();

    int a = shared[ty][tx];
    int b = shared[ty][tx + TB];
    int c = shared[ty + TB][tx];
    int d = shared[ty + TB][tx + TB];

#pragma unroll 8
    for (int k = 0; k < B; ++k)
    {
        a = min(a, shared[ty][k] + shared[k][tx]);
        b = min(b, shared[ty][k] + shared[k][tx + TB]);
        c = min(c, shared[ty + TB][k] + shared[k][tx]);
        d = min(d, shared[ty + TB][k] + shared[k][tx + TB]);
        shared[ty][tx] = a;
        shared[ty][tx + TB] = b;
        shared[ty + TB][tx] = c;
        shared[ty + TB][tx + TB] = d;
        __syncthreads();
    }
    dist[base_idx + ty * n + tx] = a;
    dist[base_idx + ty * n + (tx + TB)] = b;
    dist[base_idx + (ty + TB) * n + tx] = c;
    dist[base_idx + (ty + TB) * n + (tx + TB)] = d;
}

__global__ void phase2(int *dist, int n, int Round)
{
    if (blockIdx.x == Round)
        return;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = blockIdx.x;
    __shared__ int pivot[B][B];
    __shared__ int current[B][B];

    int pivot_addr = Round * B * n + Round * B;
    int cur_addr;
    if (blockIdx.y == 0)
        cur_addr = Round * B * n + idx * B;
    else
        cur_addr = idx * B * n + Round * B;

    pivot[ty][tx] = dist[pivot_addr + ty * n + tx];
    pivot[ty][tx + TB] = dist[pivot_addr + ty * n + (tx + TB)];
    pivot[ty + TB][tx] = dist[pivot_addr + (ty + TB) * n + tx];
    pivot[ty + TB][tx + TB] = dist[pivot_addr + (ty + TB) * n + (tx + TB)];

    current[ty][tx] = dist[cur_addr + ty * n + tx];
    current[ty][tx + TB] = dist[cur_addr + ty * n + (tx + TB)];
    current[ty + TB][tx] = dist[cur_addr + (ty + TB) * n + tx];
    current[ty + TB][tx + TB] = dist[cur_addr + (ty + TB) * n + (tx + TB)];
    __syncthreads();

    int res[4];
    res[0] = current[ty][tx];
    res[1] = current[ty][tx + TB];
    res[2] = current[ty + TB][tx];
    res[3] = current[ty + TB][tx + TB];

#pragma unroll
    for (int k = 0; k < B; ++k)
    {
        if (blockIdx.y == 0)
        {
            res[0] = min(res[0], pivot[ty][k] + current[k][tx]);
            res[1] = min(res[1], pivot[ty][k] + current[k][tx + TB]);
            res[2] = min(res[2], pivot[ty + TB][k] + current[k][tx]);
            res[3] = min(res[3], pivot[ty + TB][k] + current[k][tx + TB]);
        }
        else
        {
            res[0] = min(res[0], current[ty][k] + pivot[k][tx]);
            res[1] = min(res[1], current[ty][k] + pivot[k][tx + TB]);
            res[2] = min(res[2], current[ty + TB][k] + pivot[k][tx]);
            res[3] = min(res[3], current[ty + TB][k] + pivot[k][tx + TB]);
        }
    }
    dist[cur_addr + ty * n + tx] = res[0];
    dist[cur_addr + ty * n + (tx + TB)] = res[1];
    dist[cur_addr + (ty + TB) * n + tx] = res[2];
    dist[cur_addr + (ty + TB) * n + (tx + TB)] = res[3];
}

__global__ void phase3(int *dist, int n, int Round, int offset_y)
{
    int bx = blockIdx.x;
    int by = blockIdx.y + offset_y;

    if (bx == Round || by == Round)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int row_data[B][B];
    __shared__ int col_data[B][B];

    int row_addr = by * B * n + Round * B;
    int col_addr = Round * B * n + bx * B;
    int self_addr = by * B * n + bx * B;
    row_data[ty][tx] = dist[row_addr + ty * n + tx];
    row_data[ty][tx + TB] = dist[row_addr + ty * n + (tx + TB)];
    row_data[ty + TB][tx] = dist[row_addr + (ty + TB) * n + tx];
    row_data[ty + TB][tx + TB] = dist[row_addr + (ty + TB) * n + (tx + TB)];

    col_data[ty][tx] = dist[col_addr + ty * n + tx];
    col_data[ty][tx + TB] = dist[col_addr + ty * n + (tx + TB)];
    col_data[ty + TB][tx] = dist[col_addr + (ty + TB) * n + tx];
    col_data[ty + TB][tx + TB] = dist[col_addr + (ty + TB) * n + (tx + TB)];
    __syncthreads();

    int d[4];
    d[0] = dist[self_addr + ty * n + tx];
    d[1] = dist[self_addr + ty * n + (tx + TB)];
    d[2] = dist[self_addr + (ty + TB) * n + tx];
    d[3] = dist[self_addr + (ty + TB) * n + (tx + TB)];

#pragma unroll 8
    for (int k = 0; k < B; ++k)
    {
        int r0 = row_data[ty][k];
        int r1 = row_data[ty + TB][k];
        int c0 = col_data[k][tx];
        int c1 = col_data[k][tx + TB];
        d[0] = min(d[0], r0 + c0);
        d[1] = min(d[1], r0 + c1);
        d[2] = min(d[2], r1 + c0);
        d[3] = min(d[3], r1 + c1);
    }
    dist[self_addr + ty * n + tx] = d[0];
    dist[self_addr + ty * n + (tx + TB)] = d[1];
    dist[self_addr + (ty + TB) * n + tx] = d[2];
    dist[self_addr + (ty + TB) * n + (tx + TB)] = d[3];
}

void input(char *infile)
{
    FILE *file = fopen(infile, "rb");
    if (!file)
        exit(1);
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    padded_n = n;
    if (n % B != 0)
        padded_n = (n / B + 1) * B;
    cudaMallocHost(&Dist, (size_t)padded_n * padded_n * sizeof(int));
    for (int i = 0; i < padded_n; ++i)
    {
        for (int j = 0; j < padded_n; ++j)
        {
            if (i == j && i < n)
                Dist[i * padded_n + j] = 0;
            else
                Dist[i * padded_n + j] = INF;
        }
    }
    int pair[3];
    for (int i = 0; i < m; ++i)
    {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * padded_n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i)
    {
        fwrite(&Dist[i * padded_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
    cudaFreeHost(Dist);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
        return 1;
    input(argv[1]);

    int round = padded_n / B;
    size_t matrix_size = (size_t)padded_n * padded_n * sizeof(int);

    // Allocate GPU memory for 2 devices
    int *Dist_GPU[2];

// Use OpenMP to distribute work across 2 GPUs
#pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num();
        cudaSetDevice(id);

        // Partition rows: GPU 0 handles [0, round/2), GPU 1 handles [round/2, round)
        int num_rows = round / 2;
        int start = id * num_rows;
        if (id == 1)
            num_rows = round - start;

        cudaMalloc(&Dist_GPU[id], matrix_size);

        // Copy only the assigned rows to each GPU (bandwidth optimization)
        size_t off = (size_t)start * B * padded_n;
        size_t cnt = (size_t)num_rows * B * padded_n;

        cudaMemcpy(Dist_GPU[id] + off, Dist + off, cnt * sizeof(int), cudaMemcpyHostToDevice);

#pragma omp barrier
        dim3 block_dim(TB, TB);

        for (int r = 0; r < round; ++r)
        {
            // Determine which GPU owns the pivot block for this round
            int owner = (r < round / 2) ? 0 : 1;

            // Send pivot row to the other GPU via cudaMemcpyPeer
            if (id == owner)
            {
                size_t row_off = (size_t)r * B * padded_n;
                cudaMemcpyPeer(Dist_GPU[!id] + row_off, !id, Dist_GPU[id] + row_off, id, B * padded_n * sizeof(int));
            }

#pragma omp barrier

            phase1<<<1, block_dim>>>(Dist_GPU[id], padded_n, r);
            phase2<<<dim3(round, 2), block_dim>>>(Dist_GPU[id], padded_n, r);
            // Phase 3: each GPU computes only its assigned row blocks using offset_y
            phase3<<<dim3(round, num_rows), block_dim>>>(Dist_GPU[id], padded_n, r, start);
        }

        // Copy results back (only assigned rows)
        cudaMemcpy(Dist + off, Dist_GPU[id] + off, cnt * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(Dist_GPU[id]);
    }

    output(argv[2]);
    return 0;
}