import numpy as np
import pybind11_sparse_matrix

# Example usage

M = 5
N = M*M
V = np.asarray(np.random.random((M,M*3)), order='F')
R_extended = np.asarray(np.random.random((N+3)), order='F')
points = np.asarray(np.random.random((N*3)), order='F')

pybind11_sparse_matrix.update_R(V, R_extended, points, N)

print("Updated R_extended:")
print(R_extended)