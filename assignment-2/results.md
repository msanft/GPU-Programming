# Results for homework 2

```
Test Case #0
10-by-10 matrix
        CPU time: 0.000200ms
        GPU (16-by-16 block size): 0.000000ms
        GPU (16-by-32 block size): 0.000000ms
        GPU (32-by-16 block size): 0.000000ms
Test Case #1
100-by-1000 matrix
        CPU time: 0.139100ms
        GPU (16-by-16 block size): 0.000000ms
        GPU (16-by-32 block size): 0.000000ms
        GPU (32-by-16 block size): 0.000000ms
Test Case #2
1000-by-1000 matrix
        CPU time: 1.440000ms
        GPU (16-by-16 block size): 0.000000ms
        GPU (16-by-32 block size): 0.000000ms
        GPU (32-by-16 block size): 0.000000ms
Test Case #3
500-by-2000 matrix
        CPU time: 1.356900ms
        GPU (16-by-16 block size): 0.000000ms
        GPU (16-by-32 block size): 0.000000ms
        GPU (32-by-16 block size): 0.000000ms
Test Case #4
100-by-10000 matrix
        CPU time: 1.449500ms
        GPU (16-by-16 block size): 0.000000ms
        GPU (16-by-32 block size): 0.000000ms
        GPU (32-by-16 block size): 0.000000ms
```

As can be seen here, the GPU (NVIDIA RTX 3070) seems to be too fast for the timer used by the `helper_cuda.h` implementation.
Even with the supposedly higher-precision `GetSystemTimeAsFileTime` API of Windows, the GPU timings were 0 for the given matrix sizes.

When the matrix sizes are increased, reasonable results are found:

```
1000-by-100000 matrix
        CPU time: 131.712006ms
        GPU (16-by-16 block size): 4.000000ms
        GPU (16-by-32 block size): 53.000000ms
        GPU (32-by-16 block size): 6.000000ms
```
