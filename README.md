import numpy as np

# 1. Creating Matrices
# Initials: PBS => [16, 2, 9]
# Second letters: AAO => [1, 1, 15]
matrix1 = np.array([[16, 2, 9], [1, 1, 15]])

# Student number: 202555 => [2, 0, 2], [5, 5, 5]
matrix2 = np.array([[2, 0, 2], [5, 5, 5]])

# 2. Matrix Addition
matrix3 = matrix1 + matrix2

# 3. Scalar Multiplication
matrix4 = matrix1 * 2

# 4. Matrix Transpose
matrix5 = matrix2.T

# 5. Matrix Multiplication
for i in range(2):
for j in range(2):
matrix6[i][j]

# 6. Sum of all elements in Matrix 3
sum_matrix3 = np.sum(matrix3)

# 7. Zero Matrix
matrix7 = np.zeros((2, 3))

# Printing all matrices
print("Matrix 1:\n", matrix1)
print("\nMatrix 2:\n", matrix2)
print("\nMatrix 3 (Addition):\n", matrix3)
print("\nMatrix 4 (Scalar Multiplication):\n", matrix4)
print("\nMatrix 5 (Transpose of Matrix 2):\n", matrix5)
print("\nMatrix 6 (Matrix Multiplication):\n", matrix6)
print("\nSum of all elements in Matrix 3:", sum_matrix3)
print("\nMatrix 7 (Zero Matrix):\n", matrix7)
