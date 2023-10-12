import numpy as np
import torch
import time

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.rand(size=(3, 4)).to(device)
print(x)

data_size = 5000 
a = torch.randn(data_size, data_size)
b = torch.randn(data_size, data_size)

repetitions = 10

# CPU Benchmark
start_time = time.time()

for _ in range(repetitions):
    c_cpu = torch.mm(a, b)

end_time = time.time()
cpu_duration = end_time - start_time

# MPS (GPU) Benchmark
device = torch.device("mps") 
a_mps = a.to(device)
b_mps = b.to(device)

start_time = time.time()

for _ in range(repetitions):
    c_mps = torch.mm(a_mps, b_mps)

end_time = time.time()
mps_duration = end_time - start_time

print(f"CPU Time: {cpu_duration / repetitions} seconds per iteration")
print(f"MPS Time: {mps_duration / repetitions} seconds per iteration")
print(f"MPS is {cpu_duration/mps_duration} time faster than CPU for this task")
