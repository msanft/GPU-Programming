# Results for Homework 3

```
(base) ms@ms-gpu-test:~/gpu-programming/assignment-3$ ./a.out
Iterations: 100000000
Block width: 1, block height: 1
Thread count: 1
	ILP4 time: 644821485 ns (1.61 ns per operation)
	Non-ILP time: 274950885 ns (2.75 ns per operation)
	Memory ILP time: 11330173537 ns (28.33 ns per operation)
	Speedup ILP4 vs. non-ILP: 70.56%
	Speedup Memory ILP vs. non-ILP: -90.29%
Block width: 2, block height: 2
Thread count: 4
	ILP4 time: 552481924 ns (1.38 ns per operation)
	Non-ILP time: 275924609 ns (2.76 ns per operation)
	Memory ILP time: 11355952681 ns (28.39 ns per operation)
	Speedup ILP4 vs. non-ILP: 99.77%
	Speedup Memory ILP vs. non-ILP: -90.28%
Block width: 4, block height: 4
Thread count: 16
	ILP4 time: 554354584 ns (1.39 ns per operation)
	Non-ILP time: 275180936 ns (2.75 ns per operation)
	Memory ILP time: 11355956775 ns (28.39 ns per operation)
	Speedup ILP4 vs. non-ILP: 98.56%
	Speedup Memory ILP vs. non-ILP: -90.31%
Block width: 8, block height: 8
Thread count: 64
	ILP4 time: 554273164 ns (1.39 ns per operation)
	Non-ILP time: 275183398 ns (2.75 ns per operation)
	Memory ILP time: 11558114272 ns (28.90 ns per operation)
	Speedup ILP4 vs. non-ILP: 98.59%
	Speedup Memory ILP vs. non-ILP: -90.48%
Block width: 16, block height: 16
Thread count: 256
	ILP4 time: 1017702459 ns (2.54 ns per operation)
	Non-ILP time: 285776773 ns (2.86 ns per operation)
	Memory ILP time: 11815547953 ns (29.54 ns per operation)
	Speedup ILP4 vs. non-ILP: 12.32%
	Speedup Memory ILP vs. non-ILP: -90.33%
Block width: 32, block height: 32
Thread count: 1024
	ILP4 time: 4012226898 ns (10.03 ns per operation)
	Non-ILP time: 1003766800 ns (10.04 ns per operation)
	Memory ILP time: 33635374918 ns (84.09 ns per operation)
	Speedup ILP4 vs. non-ILP: 0.07%
	Speedup Memory ILP vs. non-ILP: -88.06%
```
