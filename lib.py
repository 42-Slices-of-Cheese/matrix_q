import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.complex128

# Single qubit basis states
ket_0 = torch.tensor([1.0, 0.0], dtype=dtype, device=device)  # |0>
ket_1 = torch.tensor([0.0, 1.0], dtype=dtype, device=device)  # |1>

# Identity Gate
I = torch.eye(2, dtype=dtype, device=device)

# Hadamard Gate (creates superposition)
H = (1 / math.sqrt(2)) * torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device)

CNOT = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=dtype, device=device)

SWAP = torch.tensor([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=dtype, device=device)
