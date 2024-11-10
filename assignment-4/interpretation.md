# Interpretation of Results for Homework 4

## Host vs. Device computations

Comparing CPU and GPU results, we can see that for a matrix size of `M=1000x500,N=500x2000` and a tile width of 32,
we get a speedup of around 3642 ms (~3648 ms on CPU, ~6 ms on GPU, no shared memory used). Using shared memory, this
number decreases by another ~1.3 ms from ~6 ms without shared memory to ~5 ms with it. This can be attributed to the
general speedup factor of GPUs and their parallel data processing, as the matrix multiplication operations are entirely
independent of each other and can be executed in parallel completely, while everything has to be done sequentially on the CPU.
Further, the GPU can benefit of optimizations like fused multiply-add (FMA) [1] when calculating the `p_value`, which might not
be available on CPUs.

## Impact of tile width

Analyzing different tile widths for the algorithm, it showed that a `TILE_WIDTH` of 32 resulted in optimal timings for the matrix multiplication.
The difference from a tile width of 16 was not dramatical at only about 12%, compared to tile widths like 8 or 4, that resulted in a performance penalty
of around 50% (200%, respectively). This is most likely due to this being the warp size (32 threads) [2], and 32 memory banks [3], maximizing the memory
throughput and minimizing the overhead of partial warps.

## Impact of matrix size

The results show that for larger matrix sizes, the GPU time increases linearly (as long as the calculations can be parallelized, presumably), while
on the CPU, the computation times seem to increase exponentially before getting infeasibly long. CPU results are about `x^2` when matrix size doubles,
while GPU timings are around `6*x`.

## Bank conflicts

Our memory access pattern should not cause any conflicts, as the different threads access consecutive elements in the same row (i.e. `tx`). All threads
in the same warp access within the same row (i.e. `ty`). So nothing should be accessed by two threads at the same time.

## References:

1: https://www.esa.informatik.tu-darmstadt.de/assets/publications/materials/2013/01_ArchitectureExplorationofHigh-PerformanceFloating-PointFusedMultiply-Add.pdf
2: https://forums.developer.nvidia.com/t/maximum-number-of-warps-and-warp-size-per-sm/234378
3: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#shared-memory-bandwidth
