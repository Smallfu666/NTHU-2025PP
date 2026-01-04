#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define B 64
const int INF = 1073741823;

int n, m;
int *Dist = NULL;

void input(char *infile);
void output(char *outfile);
void fw();
void cal(int r, int sx, int sy, int w, int h);

int main(int argc, char *argv[])
{
    if (argc < 3)
        return 1;

    input(argv[1]);
    fw();
    output(argv[2]);

    free(Dist);
    return 0;
}

void input(char *infile)
{
    FILE *file = fopen(infile, "rb");
    if (!file)
        exit(1);

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (int *)malloc((size_t)n * n * sizeof(int));

#pragma omp parallel for
    for (long long i = 0; i < (long long)n * n; ++i)
        Dist[i] = INF;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        Dist[i * n + i] = 0;

    int pair[3];
    for (int i = 0; i < m; ++i)
    {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outfile)
{
    FILE *out = fopen(outfile, "w");
    for (int i = 0; i < n; ++i)
        fwrite(&Dist[i * n], sizeof(int), n, out);
    fclose(out);
}

void fw()
{
    int round = (n + B - 1) / B;
    for (int r = 0; r < round; ++r)
    {
        cal(r, r, r, 1, 1);

        cal(r, r, 0, r, 1);                 // 左上
        cal(r, r, r + 1, round - r - 1, 1); // 右上
        cal(r, 0, r, 1, r);                 // 左下
        cal(r, r + 1, r, 1, round - r - 1); //  右下

        cal(r, 0, 0, r, r); // 左上象限 0 到 r-1 列，0 到 r-1 行)

        cal(r, 0, r + 1, round - r - 1, r);                 // 右上象限 0 到 r-1 列，r+1 到最後一行)
        cal(r, r + 1, 0, r, round - r - 1);                 // 左下象限 r+1 到最後一列，0 到 r-1 行)
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1); // 右下象限 r+1 到最後一列，r+1 到最後一行)
    }
}

void cal(int r, int sx, int sy, int w, int h)
{
    int ex = sx + h;
    int ey = sy + w;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = sx; bi < ex; ++bi)
    {
        for (int bj = sy; bj < ey; ++bj)
        {
            int istart = bi * B;
            int iend = (bi + 1) * B;
            int jstart = bj * B;
            int jend = (bj + 1) * B;

            if (iend > n)
                iend = n;
            if (jend > n)
                jend = n;

            int kstart = r * B;
            int kend = (r + 1) * B;
            if (kend > n)
                kend = n;

            for (int k = kstart; k < kend; ++k)
            {
                for (int i = istart; i < iend; ++i)
                {
                    int dik = Dist[i * n + k];
                    for (int j = jstart; j < jend; ++j)
                    {
                        int dkj = Dist[k * n + j];
                        int dij = Dist[i * n + j];
                        if (dik + dkj < dij)
                            Dist[i * n + j] = dik + dkj;
                    }
                }
            }
        }
    }
}