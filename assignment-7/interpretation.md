# Interpretation of Results for Homework 7

## Atomic-Add-only

While the CPU-based implementation takes around 30ms, the performance on the GPU using _only_ the atomic add operation even degrades by 6 ms. This is unexpected, but
could possibly be explained by accounting for the individually worse performance of the GPU cores compared to the CPU. Additionally, the CPU might benefit from better
pipelining optimizations. Also, the GPU has the overhead of always having to attain a lock when writing into the result, which might come at a performance overhead on itself,
as an additional value needs to be written.

## Cascaded approach

Utilizing the cascaded approach described in the homework description, we see a rather massiv performance increase to around 0.29 ms. This can be explained by all the optimizations
made for this implementation, which reduce the faster per-thread data structures which don't require individual locking and let us reduce the vector with only one atomic add per
individual block. Furthermore, this implementation benefits a lot from the coalesced memory accesses.
