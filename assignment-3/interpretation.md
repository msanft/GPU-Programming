# Interpretation of Results for Homework 3

## Non-ILP

For non-ILP computations, we see steady execution times of around 2.75ns per arithmetic operation
up until 1024 threads, where we see execution time to roughly quadruple. This could possibly be caused
by the IO bottleneck between host and GPU, where for each execution substantially more data needs to be
copied over.

## ILP

For the ILP4 computations, which are independent operations on static data that the GPU can parallelize, the computations are about twice as fast than the non-ILP computations up until 256 threads,
where they take about the same amount of time of about 2.75ns. I cannot explain why the ILP4 computations show the ~100% speed
increase for 1 to 64 threads. The expected result would be them taking just as long as the non-ILP computations, as with ILP4,
the 4 arithmetic operations should take exactly as long as 1 arithmetic operation in the non-ILP setup. A further speed increase
suggests that there is another factor influencing this, but I couldn't find out what that might be on my own by just Googling.
It might be caused by some other hardware optimizations allowing for better pipelining of the instructions here, but that's just speculations.
Nonetheless, with 256 to 1024 threads, we see similar times on both ILP4 and non-ILP, which shows that in these thread-count regions,
the given GPU (NVIDIA T4) cannot utilize the ILP anymore. This is expected and correlates with the task, which asked to prove
that with increasing task counts, the ILP becomes less effective.

## Memory ILP

The memory ILP, where the GPU should perform independent load and store operations on memory that it should be able to parallelize,
showed that they are about 10 times as slow as the ILP4 counterparts, indicating that the load operation is very costly. (The ILP4 part
only does a store when writing the result, not reading anything from the memory.) This is much more of a slowdown than what I had initially expected,
and I am still suspecting an error on my side here, as memory loads (especially from fast GPU memory), should not take this long.
In addition, the memory ILP does not seem to be able to utilize the oprimization(?) the ILP version uses to achieve the 2x speed increase from
1 to 64 threads, making its time consumption grow on a steady scale along the non-ILP one.
