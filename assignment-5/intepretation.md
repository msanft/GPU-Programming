# Interpretation of Results for Homework 4

## Row-to-row copy

The simple row-to-row copy approach showed the best results at almost 200 GB/s. This can be attributed to it requiring almost no computations (except for one index calculation) and
it benefiting from GPU parallelism best. However, it needs to be noted that this is not matrix transposal on itself. Thus the results need to be taken with a grain of salt.

## Naive GPU approach

This approach yields an effective bandwidth of 69 GB/s, being the worst of all GPU implementations. Compared to matrix transposal on the CPU using the same algorithm, this is still a
massive performance increase by a factor of 69, however. Compared to the shared memory GPU implementation, this approach cannot benefit from memory coalescing as much, as on the write
operations to the output matrix, there are large strides between the individual writes. For the reading of the input matrix, this should be exactly as efficient as in the row-to-row copy
approach.

## Shared memory GPU approach

This is the best "real" matrix transposal implementation on the GPU of the ones tested, having an effective bandwidth of 174 GB/s, nearing the performance of the simple copy kernel.
By writing to the shared memory blocks, a more coalescing-friendly pattern of write operations to the output matrix is achieved as both input and result are written with matching strides.
The bank conflicts are avoided by allocating `TILE_DIM x TILE_DIM+1` of shared memory. These optimizations allow the high performance of this implementation.
