#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <chrono>
#include <vector>

// write_png 函數 (與你提供的一樣，保持不變)
void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    assert(row);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{
    /* total execution time */
    auto total_start = std::chrono::steady_clock::now();

    /* MPI initialization */
    auto mpi_init_start = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto mpi_init_end = std::chrono::steady_clock::now();

    /* setup and argument parsing */
    auto setup_start = std::chrono::steady_clock::now();
    assert(argc == 9);
    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    if (!getenv("OMP_SCHEDULE"))
    {
        omp_set_schedule(omp_sched_dynamic, 1);
    }

    double dx = (right - left) / width;
    double dy = (upper - lower) / height;

    size_t total_pixels = (size_t)width * height;
    int *local_image = (int *)calloc(total_pixels, sizeof(int));
    assert(local_image);

    const double const_2_val = 2.0;
    __m128d vec_2 = _mm_load1_pd(&const_2_val);
    auto setup_end = std::chrono::steady_clock::now();

    /* computation time */ /* computation time */
    auto compute_start = std::chrono::steady_clock::now();

    // 每 rank 的 thread 計時陣列
    int T = omp_get_max_threads();
    std::vector<double> thread_ms(T, 0.0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double t0 = omp_get_wtime(); // 每個 thread 的起點（包住整個 for）

#pragma omp for schedule(dynamic, 1)
        for (int row = rank; row < height; row += size)
        {
            // === 你的 SIMD 計算（原樣）===
            size_t row_offset = (size_t)row * width;
            __m128d y_init = _mm_set1_pd(lower + (double)row * dy);

            int idx0 = 0, idx1 = 1, next_idx = 2;
            int count0 = 0, count1 = 0;

            __m128d x_init = _mm_set_pd(left + (double)idx1 * dx, left + (double)idx0 * dx);
            __m128d x = _mm_setzero_pd();
            __m128d y = _mm_setzero_pd();
            __m128d len_sq = _mm_setzero_pd();
            __m128d x_sq = _mm_setzero_pd();
            __m128d y_sq = _mm_setzero_pd();

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
                    local_image[row_offset + idx0] = count0;
                    idx0 = next_idx++;
                    if (idx0 >= width)
                        break;
                    count0 = 0;

                    double x_low = left + (double)idx0 * dx, x_high;
                    _mm_storeh_pd(&x_high, x_init);
                    x_init = _mm_set_pd(x_high, x_low);
                    x[0] = 0;
                    y[0] = 0;
                    len_sq[0] = 0;

                    if (count1 == iters || len_arr[1] >= 4.0)
                    {
                        local_image[row_offset + idx1] = count1;
                        idx1 = next_idx++;
                        if (idx1 >= width)
                            break;
                        count1 = 0;

                        _mm_storel_pd(&x_low, x_init);
                        x_high = left + (double)idx1 * dx;
                        x_init = _mm_set_pd(x_high, x_low);
                        x[1] = 0;
                        y[1] = 0;
                        len_sq[1] = 0;
                    }
                    x_sq = _mm_mul_pd(x, x);
                    y_sq = _mm_mul_pd(y, y);
                }
                else if (count1 == iters || len_arr[1] >= 4.0)
                {
                    local_image[row_offset + idx1] = count1;
                    idx1 = next_idx++;
                    if (idx1 >= width)
                        break;
                    count1 = 0;

                    double x_low;
                    _mm_storel_pd(&x_low, x_init);
                    double x_high = left + (double)idx1 * dx;
                    x_init = _mm_set_pd(x_high, x_low);
                    x[1] = 0;
                    y[1] = 0;
                    len_sq[1] = 0;
                    x_sq = _mm_mul_pd(x, x);
                    y_sq = _mm_mul_pd(y, y);
                }
            }

            if (idx1 >= width && idx0 < width)
            {
                double px = x[0], py = y[0];
                double x0 = left + (double)idx0 * dx;
                double y0 = lower + (double)row * dy;
                while (count0 < iters && (px * px + py * py) < 4.0)
                {
                    double temp = px * px - py * py + x0;
                    py = 2.0 * px * py + y0;
                    px = temp;
                    count0++;
                }
                local_image[row_offset + idx0] = count0;
            }
            else if (idx0 >= width && idx1 < width)
            {
                double px = x[1], py = y[1];
                double x0 = left + (double)idx1 * dx;
                double y0 = lower + (double)row * dy;
                while (count1 < iters && (px * px + py * py) < 4.0)
                {
                    double temp = px * px - py * py + x0;
                    py = 2.0 * px * py + y0;
                    px = temp;
                    count1++;
                }
                local_image[row_offset + idx1] = count1;
            }
            // === 計算結束 ===
        }

        double t1 = omp_get_wtime();
        thread_ms[tid] = (t1 - t0) * 1000.0; // 每個 thread 的總毫秒
    } // end parallel

    auto compute_end = std::chrono::steady_clock::now();

    /* memory allocation for result */
    auto mem_start = std::chrono::steady_clock::now();
    int *final_image = NULL;
    if (rank == 0)
    {
        final_image = (int *)malloc(total_pixels * sizeof(int));
        assert(final_image);
    }
    auto mem_end = std::chrono::steady_clock::now();

    // ========================================================================
    // ** 這是新加入的區塊 **
    // ========================================================================
    /* Synchronization (wait for slowest process) */
    // 這一步測量的就是 "Load Imbalance" 造成的等待時間
    auto sync_start = std::chrono::steady_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto sync_end = std::chrono::steady_clock::now();
    // ========================================================================

    /* MPI communication */
    // 現在所有進程都同步了，這裡測量的就是 "純粹的" 歸約通訊時間
    auto comm_start = std::chrono::steady_clock::now();
    MPI_Reduce(local_image, final_image, (int)total_pixels, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    auto comm_end = std::chrono::steady_clock::now();

    /* output file */
    auto io_start = std::chrono::steady_clock::now();
    if (rank == 0)
    {
        write_png(filename, iters, width, height, final_image);
        free(final_image);
    }
    auto io_end = std::chrono::steady_clock::now();

    /* cleanup */
    auto cleanup_start = std::chrono::steady_clock::now();
    free(local_image);
    MPI_Finalize();
    auto cleanup_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();

    /* calculate times in milliseconds */
    auto mpi_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(mpi_init_end - mpi_init_start).count();
    auto setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
    auto mem_time = std::chrono::duration_cast<std::chrono::milliseconds>(mem_end - mem_start).count();

    // ** 新增 sync_time **
    auto sync_time = std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start).count();

    auto comm_time = std::chrono::duration_cast<std::chrono::milliseconds>(comm_end - comm_start).count();
    auto io_time = std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
    auto cleanup_time = std::chrono::duration_cast<std::chrono::milliseconds>(cleanup_end - cleanup_start).count();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    /* output performance report */
    if (rank == 0)
    {
        printf("=== hw2b Performance Report ===\n");
        printf("MPI processes: %d\n", size);
        printf("Image size: %dx%d (%zu pixels)\n", width, height, total_pixels);
        printf("Iterations: %d\n", iters);
        printf("\n--- Timing Breakdown (ms) ---\n");
        printf("MPI Init:     %lld ms\n", mpi_init_time);
        printf("Setup:        %lld ms\n", setup_time);
        printf("Computation:  %lld ms\n", compute_time);
        printf("Memory:       %lld ms\n", mem_time);
        printf("ThreadTimes(ms):");
        // 在 rank==0 的報表區塊裡補一段：

        printf("ThreadTimes(ms):");
        for (int i = 0; i < (int)thread_ms.size(); ++i)
        {
            printf(" %.3f", thread_ms[i]);
        }
        printf("\n");
    }

    // ** 新增 Sync (Wait) 報告 **
    printf("Sync (Wait):  %lld ms\n", sync_time);

    printf("MPI Comm:     %lld ms\n", comm_time);
    printf("I/O Output:   %lld ms\n", io_time);
    printf("Cleanup:      %lld ms\n", cleanup_time);
    printf("Total:        %lld ms\n", total_time);
    printf("\n--- Performance Metrics ---\n");
    printf("Compute %%:     %.1f%% \n", 100.0 * compute_time / total_time);
    printf("Per pixel:    %.3f ms\n", (double)compute_time / total_pixels);
    printf("Throughput:   %.1f pixels/sec\n", 1000.0 * total_pixels / total_time);
    printf("Speedup:      %.1fx (theoretical)\n", (double)size);

    return 0;
}
