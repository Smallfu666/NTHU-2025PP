#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


#ifndef B
#define B 64
#endif

#define TB (B / 2)

const int INF = 1073741823;

int n, m, padded_n;
int *Dist = NULL;

// Phase 1: 處理對角線 Pivot Block
__global__ void phase1(int *dist, int n, int Round)
{
    // Shared Memory 大小隨 B 改變
    __shared__ int shared[B][B];

    // 計算 Pivot Block 在 Global Memory 的起始位置
    // (Round * B * n) 是 Y 軸偏移， (Round * B) 是 X 軸偏移
    int base_idx = Round * B * n + Round * B;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 1. 搬運資料 (每個 Thread 搬 4 筆)
    // 使用 TB 作為 stride
    shared[ty][tx] = dist[base_idx + ty * n + tx];
    shared[ty][tx + TB] = dist[base_idx + ty * n + (tx + TB)];
    shared[ty + TB][tx] = dist[base_idx + (ty + TB) * n + tx];
    shared[ty + TB][tx + TB] = dist[base_idx + (ty + TB) * n + (tx + TB)];

    __syncthreads();

    // 2. 計算 (Register Blocking) - Floyd-Warshall
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

        // 更新共享記憶體以供下一次 k 迭代使用
        shared[ty][tx] = a;
        shared[ty][tx + TB] = b;
        shared[ty + TB][tx] = c;
        shared[ty + TB][tx + TB] = d;

        __syncthreads();
    }

    // 3. 寫回 Global
    dist[base_idx + ty * n + tx] = a;
    dist[base_idx + ty * n + (tx + TB)] = b;
    dist[base_idx + (ty + TB) * n + tx] = c;
    dist[base_idx + (ty + TB) * n + (tx + TB)] = d;
}

// Phase 2: 處理十字區域 (Row & Col)
__global__ void phase2(int *dist, int n, int Round)
{
    if (blockIdx.x == Round)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = blockIdx.x;

    __shared__ int pivot_smem[B][B];
    __shared__ int self_smem[B][B];

    int pivot_base = Round * B * n + Round * B;
    int self_base;

    // 判斷是 Row Block 還是 Col Block
    if (blockIdx.y == 0)
    {
        self_base = Round * B * n + idx * B; // Row
    }
    else
    {
        self_base = idx * B * n + Round * B; // Col
    }

    // 1. Load Pivot Block
    pivot_smem[ty][tx] = dist[pivot_base + ty * n + tx];
    pivot_smem[ty][tx + TB] = dist[pivot_base + ty * n + (tx + TB)];
    pivot_smem[ty + TB][tx] = dist[pivot_base + (ty + TB) * n + tx];
    pivot_smem[ty + TB][tx + TB] = dist[pivot_base + (ty + TB) * n + (tx + TB)];

    // 2. Load Self Block
    self_smem[ty][tx] = dist[self_base + ty * n + tx];
    self_smem[ty][tx + TB] = dist[self_base + ty * n + (tx + TB)];
    self_smem[ty + TB][tx] = dist[self_base + (ty + TB) * n + tx];
    self_smem[ty + TB][tx + TB] = dist[self_base + (ty + TB) * n + (tx + TB)];

    __syncthreads();

    // 3. Compute
    int val00 = self_smem[ty][tx];
    int val01 = self_smem[ty][tx + TB];
    int val10 = self_smem[ty + TB][tx];
    int val11 = self_smem[ty + TB][tx + TB];

#pragma unroll
    for (int k = 0; k < B; ++k)
    {
        int p_row_k = pivot_smem[ty][k];
        int p_row_k_TB = pivot_smem[ty + TB][k];
        int p_col_k = pivot_smem[k][tx];
        int p_col_k_TB = pivot_smem[k][tx + TB];

        int s_row_k = self_smem[ty][k];
        int s_row_k_TB = self_smem[ty + TB][k];
        int s_col_k = self_smem[k][tx];
        int s_col_k_TB = self_smem[k][tx + TB];

        if (blockIdx.y == 0)
        {
            // Row Block: Pivot(row) + Self(col)
            // Pivot 在左邊，Self 在右邊
            // D[Round][i] = min(..., D[Round][k] + D[k][i])
            // D[Round][k] 是 pivot_smem[ty][k]
            // D[k][i] 是 self_smem[k][tx]
            val00 = min(val00, p_row_k + s_col_k);
            val01 = min(val01, p_row_k + s_col_k_TB);
            val10 = min(val10, p_row_k_TB + s_col_k);
            val11 = min(val11, p_row_k_TB + s_col_k_TB);
        }
        else
        {
            // Col Block: Self(row) + Pivot(col)
            val00 = min(val00, s_row_k + p_col_k);
            val01 = min(val01, s_row_k + p_col_k_TB);
            val10 = min(val10, s_row_k_TB + p_col_k);
            val11 = min(val11, s_row_k_TB + p_col_k_TB);
        }
    }

    // 4. Write Back
    dist[self_base + ty * n + tx] = val00;
    dist[self_base + ty * n + (tx + TB)] = val01;
    dist[self_base + (ty + TB) * n + tx] = val10;
    dist[self_base + (ty + TB) * n + (tx + TB)] = val11;
}

// Phase 3: 處理其餘區塊
__global__ void phase3(int *dist, int n, int Round)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 跳過 Pivot Row 和 Pivot Col
    if (bx == Round || by == Round)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 定義兩塊 Shared Memory
    __shared__ int row_smem[B][B];
    __shared__ int col_smem[B][B];

    int row_base = by * B * n + Round * B;
    int col_base = Round * B * n + bx * B;
    int self_base = by * B * n + bx * B;

    // 1. Load Dependency Blocks
    row_smem[ty][tx] = dist[row_base + ty * n + tx];
    row_smem[ty][tx + TB] = dist[row_base + ty * n + (tx + TB)];
    row_smem[ty + TB][tx] = dist[row_base + (ty + TB) * n + tx];
    row_smem[ty + TB][tx + TB] = dist[row_base + (ty + TB) * n + (tx + TB)];

    col_smem[ty][tx] = dist[col_base + ty * n + tx];
    col_smem[ty][tx + TB] = dist[col_base + ty * n + (tx + TB)];
    col_smem[ty + TB][tx] = dist[col_base + (ty + TB) * n + tx];
    col_smem[ty + TB][tx + TB] = dist[col_base + (ty + TB) * n + (tx + TB)];

    __syncthreads();

    // 2. Load Self Data to Registers
    int val00 = dist[self_base + ty * n + tx];
    int val01 = dist[self_base + ty * n + (tx + TB)];
    int val10 = dist[self_base + (ty + TB) * n + tx];
    int val11 = dist[self_base + (ty + TB) * n + (tx + TB)];

// 3. Compute
#pragma unroll
    for (int k = 0; k < B; ++k)
    {
        int r0 = row_smem[ty][k];
        int r1 = row_smem[ty + TB][k];
        int c0 = col_smem[k][tx];
        int c1 = col_smem[k][tx + TB];

        val00 = min(val00, r0 + c0);
        val01 = min(val01, r0 + c1);
        val10 = min(val10, r1 + c0);
        val11 = min(val11, r1 + c1);
    }

    // 4. Write Back
    dist[self_base + ty * n + tx] = val00;
    dist[self_base + ty * n + (tx + TB)] = val01;
    dist[self_base + (ty + TB) * n + tx] = val10;
    dist[self_base + (ty + TB) * n + (tx + TB)] = val11;
}

// =============================================================
// Host Code
// =============================================================

void input(char *infile)
{
    FILE *file = fopen(infile, "rb");
    if (!file)
        exit(1);

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // 通用 Padding 邏輯: 根據 B 補齊
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

    int *Dist_GPU;
    size_t size = (size_t)padded_n * padded_n * sizeof(int);
    cudaMalloc(&Dist_GPU, size);

    cudaMemcpy(Dist_GPU, Dist, size, cudaMemcpyHostToDevice);

    int round = padded_n / B;

    // 動態設定 Block Dimension 
    dim3 block_dim(TB, TB);

    for (int r = 0; r < round; ++r)
    {
        phase1<<<1, block_dim>>>(Dist_GPU, padded_n, r);
        phase2<<<dim3(round, 2), block_dim>>>(Dist_GPU, padded_n, r);
        phase3<<<dim3(round, round), block_dim>>>(Dist_GPU, padded_n, r);
    }

    cudaMemcpy(Dist, Dist_GPU, size, cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU);

    output(argv[2]);
    return 0;
}