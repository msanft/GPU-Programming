# Results for Homework 5

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

With `nreps=1`:

```
CPU time: 263052242 ns
Effective bandwidth: 1 GB/s
GPU time (row-to-row copy): 2021345 ns
Effective bandwidth (row-to-row copy): 197 GB/s
GPU time (naiive): 5784677 ns
Effective bandwidth (naiive): 69 GB/s
GPU time (shared memory): 2293176 ns
Effective bandwidth (shared memory): 174 GB/s
```
