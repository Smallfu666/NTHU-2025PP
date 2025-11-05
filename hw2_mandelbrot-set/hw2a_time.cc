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
#include <chrono> // 使用 std::chrono 來計時

int iters;
double left, right, lower, upper;
int width, height;
double dx, dy;
int *image;
int thread_count;
std::atomic<int> next_row;

// === 1. 新增全域陣列來儲存每個 thread 的時間 ===
// (使用 64 來匹配您 getenv 的檢查)
double thread_busy_times[64];

void compute_row(int row)
{
    // ... 您的 compute_row 函數 (完全不用變) ...
    const double const_2_val = 2.0;
    __m128d vec_2 = _mm_load1_pd(&const_2_val);

    size_t row_offset = (size_t)row * width;
    __m128d y_init = _mm_set1_pd(lower + (double)row * dy);

    int idx0 = 0, idx1 = 1, next_idx = 2;
    int count0 = 0, count1 = 0;

    if (idx0 >= width)
        return;
    if (idx1 >= width)
    {
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

    __m128d x_init = _mm_set_pd(left + (double)idx1 * dx, left + (double)idx0 * dx);
    __m128d x = _mm_setzero_pd(), y = _mm_setzero_pd();
    __m128d len_sq = _mm_setzero_pd(), x_sq = _mm_setzero_pd(), y_sq = _mm_setzero_pd();
    double len_arr[2];

    while (1)
    {
        y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_2, x), y), y_init);
        x = _mm_add_pd(_mm_sub_pd(x_sq, y_sq), x_init);
        x_sq = _mm_mul_pd(x, x);
        y_sq = _mm_mul_pd(y, y);
        len_sq = _mm_add_pd(x_sq, y_sq);
        count0++;
        count1++;
        _mm_store_pd(len_arr, len_sq);

        if (count0 == iters || len_arr[0] >= 4.0)
        {
            image[row_offset + idx0] = count0;
            idx0 = next_idx++;
            if (idx0 >= width)
                break;
            count0 = 0;
            double x_low = left + (double)idx0 * dx;
            double x_high;
            _mm_storeh_pd(&x_high, x_init);
            x_init = _mm_set_pd(x_high, x_low);
            x[0] = y[0] = len_sq[0] = 0;

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
        else if (count1 == iters || len_arr[1] >= 4.0)
        {
            image[row_offset + idx1] = count1;
            idx1 = next_idx++;
            if (idx1 >= width)
                break;
            count1 = 0;
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

// === 2. 修改 worker 函數 ===
void *worker(void *arg)
{
    // 接收 thread ID
    long tid = (long)arg;

    // Thread-local 計時器開始
    auto thread_start = std::chrono::steady_clock::now();

    while (1)
    {
        int row = next_row.fetch_add(1, std::memory_order_relaxed);
        if (row >= height)
            break;
        compute_row(row);
    }

    // Thread-local 計時器結束
    auto thread_end = std::chrono::steady_clock::now();

    // 計算此 thread 忙碌的總毫秒數 (ms)
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(thread_end - thread_start).count();

    // 存入全域陣列
    thread_busy_times[tid] = (double)duration_ms;

    return NULL;
}

int main(int argc, char **argv)
{
    assert(argc == 9);
    const char *filename = argv[1]; // unused
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    // detect CPU cores
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    thread_count = CPU_COUNT(&cpu_set);

    if (const char *env = getenv("OMP_NUM_THREADS"))
    {
        int v = atoi(env);
        if (v > 0 && v <= 64)
            thread_count = v;
    }
    if (thread_count < 1)
        thread_count = 1;
    if (thread_count > 12) // 您的程式限制最多 12
        thread_count = 12;

    dx = (right - left) / width;
    dy = (upper - lower) / height;

    image = (int *)malloc((size_t)width * height * sizeof(int));
    assert(image);
    next_row = 0;

    pthread_t threads[12];
    // 初始化計時陣列
    for (int i = 0; i < thread_count; ++i)
        thread_busy_times[i] = 0.0;

    auto start = std::chrono::steady_clock::now();

    // === 3. 修改 pthread_create ===
    for (int i = 0; i < thread_count; ++i)
        // 將 'i' (thread ID) 作為參數傳遞
        pthread_create(&threads[i], NULL, worker, (void *)(long)i);

    for (int i = 0; i < thread_count; ++i)
        pthread_join(threads[i], NULL);

    auto end = std::chrono::steady_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Threads: %d\n", thread_count);
    printf("Computation Time: %lld ms\n", compute_time);
    printf("Throughput: %.1f pixels/sec\n", 1000.0 * width * height / compute_time);

    // === 3. (續) 印出 Load Balancing 數據 ===
    printf("\n=== Load Balancing Data (ms) ===\n");
    for (int i = 0; i < thread_count; ++i)
    {
        // 印出格式 "T0_Time: 1075.00"
        printf("T%d_Time: %.2f\n", i, thread_busy_times[i]);
    }
    printf("==================================\n");
    // ======================================

    free(image);
    return 0;
}