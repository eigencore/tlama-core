import torch
import time
import matmul_cuda


def test_matmul(N):
    """Prueba de multiplicación de matrices con matrices aleatorias de tamaño N x N."""
    # Crear matrices aleatorias
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')

    # Prueba con tu implementación CUDA
    C_cuda = torch.matmul(A, B)


# Ejecutar la prueba para diferentes tamaños de matriz
test_matmul(4092)