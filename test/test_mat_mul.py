import torch
import time
import matplotlib.pyplot as plt
import matmul_cuda

def matmul_torch(A, B):
    """Multiplicación de matrices usando PyTorch estándar."""
    return torch.matmul(A, B)

def test_matmul(N):
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    # Warm-up para PyTorch
    matmul_torch(A, B)
    torch.cuda.synchronize()

    # Warm-up para CUDA
    matmul_cuda.matmul(A, B)
    torch.cuda.synchronize()

    # Eventos para sincronización
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # PyTorch
    start_event.record()
    C_torch = matmul_torch(A, B)
    end_event.record()
    torch.cuda.synchronize()  # Sincronizar la GPU
    torch_time = start_event.elapsed_time(end_event) / 1000  # Pasar a segundos

    # CUDA
    start_event.record()
    C_cuda = matmul_cuda.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    cuda_time = start_event.elapsed_time(end_event) / 1000

    # Calcular errores
    max_error = torch.max(torch.abs(C_cuda - C_torch)).item()
    relative_error = torch.mean(torch.abs(C_cuda - C_torch) / torch.abs(C_torch)).item()

    assert torch.allclose(C_cuda, C_torch, atol=1e-3), "Los resultados no coinciden"

    # Calcular FLOPS
    flops = 2 * N ** 3
    torch_flops = flops / torch_time
    cuda_flops = flops / cuda_time

    return {
        "N": N,
        "torch_time": torch_time,
        "cuda_time": cuda_time,
        "speedup": torch_time / cuda_time,
        "max_error": max_error,
        "relative_error": relative_error,
        "torch_flops": torch_flops,
        "cuda_flops": cuda_flops,
    }


# Lista de tamaños de matriz a probar
sizes = [256, 512, 1024, 2048, 4096]

# Ejecutar pruebas y almacenar resultados
results = []
for N in sizes:
    print(f"Probando tamaño de matriz: {N}x{N}")
    result = test_matmul(N)
    results.append(result)
    print(f"Tiempo de PyTorch: {result['torch_time']:.6f} segundos")
    print(f"Tiempo de CUDA: {result['cuda_time']:.6f} segundos")
    print(f"Speedup: {result['speedup']:.2f}x")
    print(f"Error absoluto máximo: {result['max_error']:.6f}")
    print(f"Error relativo: {result['relative_error']:.6f}")
    print(f"FLOPS de PyTorch: {result['torch_flops'] / 1e9:.2f} GFLOPS")
    print(f"FLOPS de CUDA: {result['cuda_flops'] / 1e9:.2f} GFLOPS")
    print("-" * 40)

# Extraer datos para gráficas
Ns = [result["N"] for result in results]
torch_times = [result["torch_time"] for result in results]
cuda_times = [result["cuda_time"] for result in results]
speedups = [result["speedup"] for result in results]
torch_flops = [result["torch_flops"] / 1e9 for result in results]  # En GFLOPS
cuda_flops = [result["cuda_flops"] / 1e9 for result in results]    # En GFLOPS

# Gráfica 1: Tiempos de ejecución
plt.figure(figsize=(10, 6))
plt.plot(Ns, torch_times, marker='o', label="PyTorch")
plt.plot(Ns, cuda_times, marker='o', label="CUDA")
plt.xlabel("Tamaño de la matriz (N)")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Comparación de tiempos de ejecución")
plt.legend()
plt.grid(True)
plt.savefig("tiempos.png")  # Guardar la gráfica
plt.close()

# Gráfica 2: Speedup
plt.figure(figsize=(10, 6))
plt.plot(Ns, speedups, marker='o', label="Speedup (PyTorch / CUDA)")
plt.xlabel("Tamaño de la matriz (N)")
plt.ylabel("Speedup")
plt.title("Speedup de CUDA vs PyTorch")
plt.legend()
plt.grid(True)
plt.savefig("speedup.png")  # Guardar la gráfica
plt.close()

# Gráfica 3: FLOPS
plt.figure(figsize=(10, 6))
plt.plot(Ns, torch_flops, marker='o', label="PyTorch")
plt.plot(Ns, cuda_flops, marker='o', label="CUDA")
plt.xlabel("Tamaño de la matriz (N)")
plt.ylabel("FLOPS (GFLOPS)")
plt.title("Comparación de FLOPS alcanzados")
plt.legend()
plt.grid(True)
plt.savefig("flops.png")  # Guardar la gráfica
plt.close()