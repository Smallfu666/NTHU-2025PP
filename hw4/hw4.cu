#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

void input(char *input_filename);
void output(char *output_filename);

double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

__global__ void flash_attention_kernel(float *q, float *k, float *v, float *o, int N, int d, int B)
{
    // 設定 block size
    int br = 32, bc = 32;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 算出 batch 的 offset
    int batch_offset = bx * N * d;
    float *q_data = q + batch_offset;
    float *k_data = k + batch_offset;
    float *v_data = v + batch_offset;
    float *o_data = o + batch_offset;

    // 宣告 shared memory
    __shared__ float qi[32][64];
    __shared__ float kj[32][64];
    __shared__ float vj[32][64];
    __shared__ float sij[32][32];
    __shared__ float pij[32][32];
    __shared__ float oi[32][64];
    __shared__ float mi[32];
    __shared__ float li[32];

    if (ty < br)
    {
        for (int col = tx; col < d; col += 32)
        {
            oi[ty][col] = 0.0f;
        }
        if (tx == 0)
        {
            mi[ty] = -FLT_MAX;
            li[ty] = 0.0f;
        }
    }
    __syncthreads();

    // 把 Q 讀進 shared memory
    int qi_row = by * br + ty;
    if (qi_row < N)
    {
        for (int col = tx; col < d; col += 32)
        {
            qi[ty][col] = q_data[qi_row * d + col];
        }
    }
    else
    {
        for (int col = tx; col < d; col += 32)
        {
            qi[ty][col] = 0.0f;
        }
    }
    __syncthreads();

    int tc = (N + bc - 1) / bc;
    float scalar = 1.0f / sqrtf((float)d);

    // 跑迴圈計算
    for (int j = 0; j < tc; j++)
    {
        // 讀取 K 和 V
        int kj_row = j * bc + ty;
        if (kj_row < N)
        {
            for (int col = tx; col < d; col += 32)
            {
                kj[ty][col] = k_data[kj_row * d + col];
                vj[ty][col] = v_data[kj_row * d + col];
            }
        }
        else
        {
            for (int col = tx; col < d; col += 32)
            {
                kj[ty][col] = 0.0f;
                vj[ty][col] = 0.0f;
            }
        }
        __syncthreads();

        // 算 QK^T
        float s_val = 0.0f;
        for (int k = 0; k < d; k++)
        {
            s_val += qi[ty][k] * kj[tx][k];
        }
        s_val *= scalar;
        sij[ty][tx] = s_val;
        __syncthreads();

        // 找 max
        float mij = -FLT_MAX;
        if (tx < bc)
        {
            mij = sij[ty][tx];
        }
        for (int offset = 16; offset > 0; offset /= 2)
        {
            mij = fmaxf(mij, __shfl_xor_sync(0xffffffff, mij, offset));
        }
        mij = __shfl_sync(0xffffffff, mij, 0);

        // 算 exp
        float pij_val = expf(sij[ty][tx] - mij);
        pij[ty][tx] = pij_val;
        __syncthreads();

        float lij = pij_val;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            lij += __shfl_xor_sync(0xffffffff, lij, offset);
        }
        lij = __shfl_sync(0xffffffff, lij, 0);

        // 更新 mi, li
        float mi_old = mi[ty];
        float li_old = li[ty];
        float mi_new = fmaxf(mi_old, mij);
        float li_new = expf(mi_old - mi_new) * li_old + expf(mij - mi_new) * lij;

        if (tx == 0)
        {
            mi[ty] = mi_new;
            li[ty] = li_new;
        }
        __syncthreads();

        float alpha = expf(mi_old - mi_new);
        float beta = expf(mij - mi_new);

        // 算 O
        for (int col = tx; col < d; col += 32)
        {
            float pv = 0.0f;
            for (int t = 0; t < bc; t++)
            {
                pv += pij[ty][t] * vj[t][col];
            }
            oi[ty][col] = alpha * oi[ty][col] + beta * pv;
        }
        __syncthreads();
    }

    float li_inv = 1.0f / li[ty];
    for (int col = tx; col < d; col += 32)
    {
        oi[ty][col] *= li_inv;
    }
    __syncthreads();

    // 寫回 global memory
    if (qi_row < N)
    {
        for (int col = tx; col < d; col += 32)
        {
            o_data[qi_row * d + col] = oi[ty][col];
        }
    }
}

int main(int argc, char *argv[])
{
    input(argv[1]);

    float *d_Q, *d_K, *d_V, *d_O;
    size_t size = (size_t)B * N * d * sizeof(float);

    // allocate device memory
    cudaMalloc((void **)&d_Q, size);
    cudaMalloc((void **)&d_K, size);
    cudaMalloc((void **)&d_V, size);
    cudaMalloc((void **)&d_O, size);

    // copy data to device
    cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);

    double start = getTimeStamp();

    // 設定 grid 和 block 大小
    dim3 blockDim(32, 32);
    dim3 gridDim(B, (N + 31) / 32);

    // 執行 kernel
    flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, d, B);

    cudaDeviceSynchronize();

    double end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    // 把結果 copy 回來
    cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost);

    output(argv[2]);

    // free memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return 0;
}

void input(char *input_filename)
{
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc((size_t)B * N * d * sizeof(float));
    K = (float *)malloc((size_t)B * N * d * sizeof(float));
    V = (float *)malloc((size_t)B * N * d * sizeof(float));
    O = (float *)malloc((size_t)B * N * d * sizeof(float));

    for (int i = 0; i < B; i++)
    {
        fread(Q + (size_t)i * N * d, sizeof(float), N * d, file);
        fread(K + (size_t)i * N * d, sizeof(float), N * d, file);
        fread(V + (size_t)i * N * d, sizeof(float), N * d, file);
    }
    memset(O, 0, (size_t)B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename)
{
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), (size_t)B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}
