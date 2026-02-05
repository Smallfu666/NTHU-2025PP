# High-Performance Parallel Algorithms Implementation

Professional portfolio of parallel computing work focused on performance optimization, memory hierarchy awareness, and scalable execution across CPU, GPU, and distributed environments.

![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=c%2B%2B&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![MPI](https://img.shields.io/badge/MPI-00427E)
![OpenMP](https://img.shields.io/badge/OpenMP-007ACC)

## Project Portfolio

| Project | Key Technique | Tech Stack | Description |
| --- | --- | --- | --- |
| `hw1_odd-even-sort` | Odd-even transposition | MPI, C++ | Distributed odd-even sorting with message-passing coordination. |
| `hw2_mandelbrot-set` | Work partitioning | Pthreads, MPI, OpenMP, C++ | Parallel Mandelbrot generation with threading and hybrid MPI/OpenMP variants. |
| `hw3` | Blocked APSP | CUDA, OpenMP, C++ | Blocked Floyd-Warshall with CPU and GPU implementations. |
| `hw4` | GPU kernel optimization | CUDA, C++ | CUDA-based parallel implementation for assignment-specific workload. |
| `hw5` | Communication tuning | UCX | Configuration and diff for high-performance communication setup. |
| `lab3` | Image processing | CUDA, OpenACC, C++ | Sobel filter and MNIST processing across GPU and accelerator models. |
| `lab4` | Shared-memory tiling | CUDA, C++ | Optimized Sobel filter using shared memory and tile-based access. |

## Performance Optimization Deep Dive

### lab4 (CUDA Sobel Filter)
The CUDA Sobel filter is fast because it stages input tiles in shared memory, reducing repeated global memory loads for overlapping convolution windows. This increases data reuse, minimizes global memory traffic, and raises computational intensity by performing more arithmetic per byte fetched. Coalesced global reads and shared-memory reuse keep the kernel bandwidth-efficient.

### hw3 (Blocked Floyd-Warshall)
The blocked Floyd-Warshall implementation uses cache blocking to improve locality and reduce cache misses. By operating on tiles, it keeps working sets in shared memory (GPU) or cache (CPU), cutting global memory access and improving arithmetic intensity across phases. The structure also enables parallel execution with predictable memory access patterns.

## Build & Run
- Makefile-based builds are provided in `hw1_odd-even-sort/`, `hw2_mandelbrot-set/`, and `hw3/` (use `make -C <dir>`).
- CUDA targets require `nvcc` and a compatible NVIDIA toolchain.
- MPI/OpenMP builds assume `mpicc`/`mpicxx` and OpenMP-enabled compilers in the environment.
