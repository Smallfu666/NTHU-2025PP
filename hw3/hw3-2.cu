#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define B 64

#define TB 32
const int INF = INT_MAX / 2;

int n, m, padded_n;
int *Dist = NULL;

// Phase 1: pivot 區塊
__global__ void phase1(int *dist, int n, int Round)
{
    __shared__ int shared[B][B];

    int base_idx = Round * B * n + Round * B;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 載入到 shared memory
    shared[ty][tx] = dist[base_idx + ty * n + tx];
    shared[ty][tx + TB] = dist[base_idx + ty * n + (tx + TB)];
    shared[ty + TB][tx] = dist[base_idx + (ty + TB) * n + tx];
    shared[ty + TB][tx + TB] = dist[base_idx + (ty + TB) * n + (tx + TB)];

    __syncthreads();

    // Floyd-Warshall
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

    // 寫回
    dist[base_idx + ty * n + tx] = a;
    dist[base_idx + ty * n + (tx + TB)] = b;
    dist[base_idx + (ty + TB) * n + tx] = c;
    dist[base_idx + (ty + TB) * n + (tx + TB)] = d;
}

// Phase 2: 處理同列與同欄的區塊
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
    {
        cur_addr = Round * B * n + idx * B; // row
    }
    else
    {
        cur_addr = idx * B * n + Round * B; // col
    }

    pivot[ty][tx] = dist[pivot_addr + ty * n + tx];
    pivot[ty][tx + TB] = dist[pivot_addr + ty * n + (tx + TB)];
    pivot[ty + TB][tx] = dist[pivot_addr + (ty + TB) * n + tx];
    pivot[ty + TB][tx + TB] = dist[pivot_addr + (ty + TB) * n + (tx + TB)];

    current[ty][tx] = dist[cur_addr + ty * n + tx];
    current[ty][tx + TB] = dist[cur_addr + ty * n + (tx + TB)];
    current[ty + TB][tx] = dist[cur_addr + (ty + TB) * n + tx];
    current[ty + TB][tx + TB] = dist[cur_addr + (ty + TB) * n + (tx + TB)];

    __syncthreads();

    int a = current[ty][tx];
    int b = current[ty][tx + TB];
    int c = current[ty + TB][tx];
    int d = current[ty + TB][tx + TB];

#pragma unroll
    for (int k = 0; k < B; ++k)
    {
        if (blockIdx.y == 0)
        {
            // Row Block: 需要 Pivot 的 Row (pivot[ty][k]) + 自己的 Col (current[k][tx])
            a = min(a, pivot[ty][k] + current[k][tx]);
            b = min(b, pivot[ty][k] + current[k][tx + TB]);
            c = min(c, pivot[ty + TB][k] + current[k][tx]);
            d = min(d, pivot[ty + TB][k] + current[k][tx + TB]);
        }
        else
        {
            // Col Block: 需要 自己的 Row (current[ty][k]) + Pivot 的 Col (pivot[k][tx])
            a = min(a, current[ty][k] + pivot[k][tx]);
            b = min(b, current[ty][k] + pivot[k][tx + TB]);
            c = min(c, current[ty + TB][k] + pivot[k][tx]);
            d = min(d, current[ty + TB][k] + pivot[k][tx + TB]);
        }
    }

    dist[cur_addr + ty * n + tx] = a;
    dist[cur_addr + ty * n + (tx + TB)] = b;
    dist[cur_addr + (ty + TB) * n + tx] = c;
    dist[cur_addr + (ty + TB) * n + (tx + TB)] = d;
}

// Phase 3
__global__ void phase3(int *dist, int n, int Round)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

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

    int a = dist[self_addr + ty * n + tx];
    int b = dist[self_addr + ty * n + (tx + TB)];
    int c = dist[self_addr + (ty + TB) * n + tx];
    int d = dist[self_addr + (ty + TB) * n + (tx + TB)];

#pragma unroll 8
    for (int k = 0; k < B; ++k)
    {
        int r0 = row_data[ty][k];
        int r1 = row_data[ty + TB][k];
        int c0 = col_data[k][tx];
        int c1 = col_data[k][tx + TB];

        a = min(a, r0 + c0);
        b = min(b, r0 + c1);
        c = min(c, r1 + c0);
        d = min(d, r1 + c1);
    }

    dist[self_addr + ty * n + tx] = a;
    dist[self_addr + ty * n + (tx + TB)] = b;
    dist[self_addr + (ty + TB) * n + tx] = c;
    dist[self_addr + (ty + TB) * n + (tx + TB)] = d;
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
    {
        padded_n = (n / B + 1) * B;
    }

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

    // GPU 記憶體配置
    int *Dist_GPU;
    size_t size = (size_t)padded_n * padded_n * sizeof(int);
    cudaMalloc(&Dist_GPU, size);

    // 複製資料到 GPU
    cudaMemcpy(Dist_GPU, Dist, size, cudaMemcpyHostToDevice);

    // 執行 Blocked Floyd-Warshall
    int round = padded_n / B;
    dim3 block_dim(TB, TB);

    for (int r = 0; r < round; ++r)
    {
        phase1<<<1, block_dim>>>(Dist_GPU, padded_n, r);
        phase2<<<dim3(round, 2), block_dim>>>(Dist_GPU, padded_n, r);
        phase3<<<dim3(round, round), block_dim>>>(Dist_GPU, padded_n, r);
    }

    cudaMemcpy(Dist, Dist_GPU, size, cudaMemcpyDeviceToHost);

    // 釋放 GPU 記憶體
    cudaFree(Dist_GPU);

    output(argv[2]);
    return 0;
}
