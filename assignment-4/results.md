# Results for Homework 4

Algorithms were verified to produce the same results with smaller matrices:

```
Tile width: 32
Matrix sizes:
	M: 1000 x 500
	N: 500 x 2000
	Result: 1000 x 2000
Timings:
	CPU time: 3647922462 ns
	GPU time: 6432476 ns
	GPU (shared memory) time: 4723767 ns
```

Then, different tile widths were tested:

```
Tile width: 4
Matrix sizes:
	M: 10000 x 5000
	N: 5000 x 20000
	Result: 10000 x 20000
Timings:
	GPU time: 13923806445 ns
	GPU (shared memory) time: 14069124829 ns

Tile width: 8
Matrix sizes:
	M: 10000 x 5000
	N: 5000 x 20000
	Result: 10000 x 20000
Timings:
	GPU time: 6309641162 ns
	GPU (shared memory) time: 4645482249 ns

Tile width: 16
Matrix sizes:
	M: 10000 x 5000
	N: 5000 x 20000
	Result: 10000 x 20000
Timings:
	GPU time: 4579052697 ns
	GPU (shared memory) time: 2881017286 ns

Tile width: 32
Matrix sizes:
	M: 10000 x 5000
	N: 5000 x 20000
	Result: 10000 x 20000
Timings:
	GPU time: 4082881105 ns
	GPU (shared memory) time: 2261937600 ns
```

Using a larger tile width throws errors of the following type, `...` corresponding with the size chosen:

```
ptxas error   : Entry function '_Z32_multiply_matrices_gpu_sharedmemPfS_S_jjj' uses too much shared data (... bytes, 0xc000 max)
```

Then, the speedup factor on different matrix sizes is tested:

```
Tile width: 32
Matrix sizes:
	M: 500 x 250
	N: 250 x 1000
	Result: 500 x 1000
Timings:
	CPU time: 387775590 ns
	GPU time: 1046632 ns
	Speedup: 370.50

Tile width: 32
Matrix sizes:
	M: 1000 x 500
	N: 500 x 2000
	Result: 1000 x 2000
Timings:
	CPU time: 3667016321 ns
	GPU time: 6152126 ns
	Speedup: 596.06

Tile width: 32
Matrix sizes:
	M: 2000 x 1000
	N: 1000 x 4000
	Result: 2000 x 4000
Timings:
	CPU time: 26654155931 ns
	GPU time: 49553541 ns
	Speedup: 537.89
```
