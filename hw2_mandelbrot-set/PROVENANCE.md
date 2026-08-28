# Provenance

This directory was consolidated from a standalone repo:
`Smallfu666/NTHU-2025PP-Mandelbrot-Set` (commit `4806044`, 2025-11-05 18:25 +0800,
"Initial commit: Add Mandelbrot Set implementation with pthread and MPI+OpenMP").

The standalone repo was deleted on 2026-08-29 after its content was verified
identical (or superseded) here. The files in this directory are the newer
revision (commit `e3aebf1`, 2026-11-06 20:43 +0800, "Update comments").

## One intentional difference (`hw2a.cc`)

The standalone repo's `hw2a.cc` had an `OMP_NUM_THREADS` environment-variable
override for the thread count; the consolidated version drops it in favor of
using `CPU_COUNT(&cpu_set)` directly. The dropped block (kept here for the
record) was:

```c
    thread_count = CPU_COUNT(&cpu_set);
    if (const char *env_omp = getenv("OMP_NUM_THREADS"))
    {
        int v = atoi(env_omp);
        if (v > 0)
            thread_count = v;
    }
```

This was a benchmarking convenience and was deliberately removed in the
consolidated revision.
