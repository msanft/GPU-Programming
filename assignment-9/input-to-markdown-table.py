import re

def parse_benchmark_data(input_text):
    vector_sizes = []
    cpu_times = []
    gpu_times = []

    lines = input_text.strip().split('\n')

    current_size = None
    for line in lines:
        if 'Vector size:' in line:
            current_size = int(re.search(r'Vector size: (\d+)', line).group(1))
            vector_sizes.append(current_size)
        elif 'CPU time:' in line:
            cpu_time = float(re.search(r'CPU time: (\d+)', line).group(1))
            cpu_times.append(cpu_time / 1000000)  # Convert ns to ms
        elif 'GPU time:' in line:
            gpu_time = float(re.search(r'GPU time: (\d+)', line).group(1))
            gpu_times.append(gpu_time / 1000000)  # Convert ns to ms

    print("| Vector Size | CPU Time (ms) | GPU Time (ms) | Speed-up |")
    print("|------------|--------------|--------------|----------|")

    for size, cpu, gpu in zip(vector_sizes, cpu_times, gpu_times):
        speedup = cpu / gpu
        print(f"| {size:,} | {cpu:.3f} | {gpu:.3f} | {speedup:.2f}x |")

benchmark_data = """
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
"""

parse_benchmark_data(benchmark_data)
