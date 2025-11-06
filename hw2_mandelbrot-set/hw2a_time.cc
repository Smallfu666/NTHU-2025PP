#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <emmintrin.h>
#include <chrono> // 用 std::chrono 來做精確計時

int iters;
double left, right, lower, upper;
int width, height;
double dx, dy;
int *image;
int thread_count;
std::atomic<int> next_row;

// 建立全域陣列來記錄每個執行緒的工作時間
// 用 64 是因為 getenv 檢查時的上限
double thread_busy_times[64];

void compute_row(int row)
{
    // SSE2 向量化計算單一行的 Mandelbrot 集合
    const double const_2_val = 2.0;
    __m128d vec_2 = _mm_load1_pd(&const_2_val);

    size_t row_offset = (size_t)row * width;
    __m128d y_init = _mm_set1_pd(lower + (double)row * dy);

    int idx0 = 0, idx1 = 1, next_idx = 2;
    int count0 = 0, count1 = 0;

    // 邊界條件檢查
    if (idx0 >= width)
        return;
    if (idx1 >= width)
    {
        // 只有一個像素時，用標準方法計算
        double x0 = left + (double)idx0 * dx;
        double y0 = lower + (double)row * dy;
        double x = 0.0, y = 0.0;
        int count = 0;
        while (count < iters && (x * x + y * y) < 4.0)
        {
            double temp = x * x - y * y + x0;
            y = 2.0 * x * y + y0;
            x = temp;
            count++;
        }
        image[row_offset + idx0] = count;
        return;
    }

    // 設定 SSE2 向量，同時處理兩個像素 [idx1_x, idx0_x]
    __m128d x_init = _mm_set_pd(left + (double)idx1 * dx, left + (double)idx0 * dx);
    __m128d x = _mm_setzero_pd(), y = _mm_setzero_pd();
    __m128d len_sq = _mm_setzero_pd(), x_sq = _mm_setzero_pd(), y_sq = _mm_setzero_pd();
    double len_arr[2];

    // 動態 2-pixel 管線的主迴圈
    while (1)
    {
        // Mandelbrot 迭代：z = z² + c
        y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_2, x), y), y_init);
        x = _mm_add_pd(_mm_sub_pd(x_sq, y_sq), x_init);
        x_sq = _mm_mul_pd(x, x);
        y_sq = _mm_mul_pd(y, y);
        len_sq = _mm_add_pd(x_sq, y_sq);
        count0++;
        count1++;
        _mm_store_pd(len_arr, len_sq);

        // 檢查第一個像素是否收斂
        if (count0 == iters || len_arr[0] >= 4.0)
        {
            image[row_offset + idx0] = count0;
            idx0 = next_idx++;
            if (idx0 >= width)
                break;
            count0 = 0;
            // 動態更新向量：保留高位，更新低位
            double x_low = left + (double)idx0 * dx;
            double x_high;
            _mm_storeh_pd(&x_high, x_init);
            x_init = _mm_set_pd(x_high, x_low);
            x[0] = y[0] = len_sq[0] = 0;

            // 檢查第二個像素是否也同時收斂
            if (count1 == iters || len_arr[1] >= 4.0)
            {
                image[row_offset + idx1] = count1;
                idx1 = next_idx++;
                if (idx1 >= width)
                    break;
                count1 = 0;
                _mm_storel_pd(&x_low, x_init);
                x_high = left + (double)idx1 * dx;
                x_init = _mm_set_pd(x_high, x_low);
                x[1] = y[1] = len_sq[1] = 0;
            }
            x_sq = _mm_mul_pd(x, x);
            y_sq = _mm_mul_pd(y, y);
        }
        // 只有第二個像素收斂的情況
        else if (count1 == iters || len_arr[1] >= 4.0)
        {
            image[row_offset + idx1] = count1;
            idx1 = next_idx++;
            if (idx1 >= width)
                break;
            count1 = 0;
            // 動態更新向量：保留低位，更新高位
            double x_low;
            _mm_storel_pd(&x_low, x_init);
            double x_high = left + (double)idx1 * dx;
            x_init = _mm_set_pd(x_high, x_low);
            x[1] = y[1] = len_sq[1] = 0;
            x_sq = _mm_mul_pd(x, x);
            y_sq = _mm_mul_pd(y, y);
        }
    }
}

void *worker(void *arg)
{
    // 取得執行緒 ID
    long tid = (long)arg;

    // 記錄這個執行緒的開始時間
    auto thread_start = std::chrono::steady_clock::now();

    while (1)
    {
        // 原子操作：取得下一個要處理的 row
        int row = next_row.fetch_add(1, std::memory_order_relaxed);
        if (row >= height)
            break;
        compute_row(row);
    }

    // 記錄這個執行緒的結束時間
    auto thread_end = std::chrono::steady_clock::now();

    // 計算這個執行緒總共工作了多少毫秒
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(thread_end - thread_start).count();

    // 儲存到全域陣列，用於負載平衡分析
    thread_busy_times[tid] = (double)duration_ms;

    return NULL;
}

int main(int argc, char **argv)
{
    assert(argc == 9);
    const char *filename = argv[1]; // 未使用，只是為了相容性
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    // 偵測系統 CPU 核心數
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    thread_count = CPU_COUNT(&cpu_set);

    // 檢查環境變數來覆蓋執行緒數量
    if (const char *env = getenv("OMP_NUM_THREADS"))
    {
        int v = atoi(env);
        if (v > 0 && v <= 64)
            thread_count = v;
    }
    if (thread_count < 1)
        thread_count = 1;
    if (thread_count > 12) // 限制最多 12 個執行緒
        thread_count = 12;

    dx = (right - left) / width;
    dy = (upper - lower) / height;

    image = (int *)malloc((size_t)width * height * sizeof(int));
    assert(image);
    next_row = 0;

    pthread_t threads[12];
    // 初始化所有執行緒的計時記錄
    for (int i = 0; i < thread_count; ++i)
        thread_busy_times[i] = 0.0;

    auto start = std::chrono::steady_clock::now();

    // 建立執行緒，並傳遞執行緒 ID 作為參數
    for (int i = 0; i < thread_count; ++i)
        pthread_create(&threads[i], NULL, worker, (void *)(long)i);

    // 等待所有執行緒完成
    for (int i = 0; i < thread_count; ++i)
        pthread_join(threads[i], NULL);

    auto end = std::chrono::steady_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Threads: %d\n", thread_count);
    printf("Computation Time: %lld ms\n", compute_time);
    printf("Throughput: %.1f pixels/sec\n", 1000.0 * width * height / compute_time);

    printf("\n=== Load Balancing Data (ms) ===\n");
    for (int i = 0; i < thread_count; ++i)
    {
        printf("T%d_Time: %.2f\n", i, thread_busy_times[i]);
    }
    printf("==================================\n");

    free(image);
    return 0;
}