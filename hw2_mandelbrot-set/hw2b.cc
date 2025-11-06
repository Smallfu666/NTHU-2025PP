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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

#pragma omp parallel for schedule(dynamic, 1)
    for (int row = rank; row < height; row += size)
    {
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
            // 進行 Mandelbrot 計算的 SSE2 向量化步驟
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

                double x_low = left + (double)idx0 * dx;
                double x_high;
                _mm_storeh_pd(&x_high, x_init);
                x_init = _mm_set_pd(x_high, x_low);
                x[0] = 0;
                y[0] = 0;
                len_sq[0] = 0;

                /* check if channel 1 also finished */
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
    }

    int *final_image = NULL;
    if (rank == 0)
    {
        final_image = (int *)malloc(total_pixels * sizeof(int));
        assert(final_image);
    }

    MPI_Reduce(local_image, final_image, (int)total_pixels, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        write_png(filename, iters, width, height, final_image);
        free(final_image);
    }

    free(local_image);
    MPI_Finalize();
    return 0;
}