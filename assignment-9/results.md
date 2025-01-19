# Results for Homework 9

`nvidia-smi` output:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:04.0 Off |                    0 |
| N/A   34C    P8             14W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Results of the prefix-sum calculation:

```sh
GPU: Tesla T4
SM Count: 40
Vector size: 100000
	CPU time: 540556 ns
	GPU time: 133778 ns
Vector size: 200000
	CPU time: 1048529 ns
	GPU time: 159676 ns
Vector size: 300000
	CPU time: 914969 ns
	GPU time: 219394 ns
Vector size: 400000
	CPU time: 1229365 ns
	GPU time: 254060 ns
Vector size: 500000
	CPU time: 2657032 ns
	GPU time: 352911 ns
Vector size: 600000
	CPU time: 1843555 ns
	GPU time: 400244 ns
Vector size: 700000
	CPU time: 3227555 ns
	GPU time: 382315 ns
Vector size: 800000
	CPU time: 3212307 ns
	GPU time: 473108 ns
Vector size: 900000
	CPU time: 3912903 ns
	GPU time: 501087 ns
Vector size: 1000000
	CPU time: 4686346 ns
	GPU time: 605038 ns
```

This leaves us with the following table for the execution times:


| Vector Size | CPU Time (ms) | GPU Time (ms) | Speed-up |
|-------------|---------------|---------------|----------|
| 100,000     | 0.541         | 0.134         | 4.04x    |
| 200,000     | 1.049         | 0.160         | 6.57x    |
| 300,000     | 0.915         | 0.219         | 4.17x    |
| 400,000     | 1.229         | 0.254         | 4.84x    |
| 500,000     | 2.657         | 0.353         | 7.53x    |
| 600,000     | 1.844         | 0.400         | 4.61x    |
| 700,000     | 3.228         | 0.382         | 8.44x    |
| 800,000     | 3.212         | 0.473         | 6.79x    |
| 900,000     | 3.913         | 0.501         | 7.81x    |
| 1,000,000   | 4.686         | 0.605         | 7.75x    |

Overall, we see a substantial performance increase of our GPU program compared to the sequential CPU one.
While the CPU execution time goes up linearly with the vector size, capping out at about 4.7 milliseconds
for the largest input vector, the GPU execution time increases much less drastically, leaving us with an
even higher speed-up factor of 7.75x for the 1-million-input vector.

The GPU speed-up slightly decreases with larger inputs. This might be attributed to memory conflicts, which
are not particularly addressed in this program.
