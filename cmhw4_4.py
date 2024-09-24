import numpy as np
N = 4

k = 1
m = 1
# Step 1: Define the matrix (example: a 3x3 matrix)
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i==j:
            K[i,j] = 2
        elif j==i+1:
            K[i,j] = -1
        elif j==i-1:
            K[i,j] = -1
        elif i==0 and j==N-1:
            K[i,j] = -1
        elif j==0 and i==N-1:
            K[i,j] = -1
K = K * k

M = np.eye(N) * m
print(M)
# Step 2: Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(K)

# Step 3: Display the eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Step 4: If you're interested in eigenfrequencies (assuming a physical context)
# Eigenfrequencies are typically taken as the square root of the eigenvalues
eigenfrequencies = np.sqrt(np.abs(eigenvalues))
print("\nEigenfrequencies:")
print(eigenfrequencies)
