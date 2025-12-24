import numpy as np

def recursive_determinant(matrix, indent=0):
    n = len(matrix)
    prefix = "  " * indent
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return val

    det = 0
    print(f"{prefix}Calculating det of {n}x{n} matrix...")
    for j in range(n):
        # Sign calculation (-1)^(i+j)
        sign = (-1) ** j
        # Cofactor: remove row 0 and column j
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        
        term = sign * matrix[0][j] * recursive_determinant(sub_matrix, indent + 1)
        det += term
    return det

def lu_decomposition_step_by_step(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    
    print("Starting LU Decomposition:")
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] = U[j] - factor * U[i]
            print(f"Step: Eliminating row {j} using row {i} with factor {factor:.2f}")
    
    print("L matrix:\n", L)
    print("U matrix:\n", U)
    return L, U

def svd_via_evd(A):
    print("\n--- Starting SVD via EVD ---")
    # 1. Solve for A^T A
    ata = A.T @ A
    print("1. Computed A^T * A:\n", ata)
    
    # 2. Eigen-decomposition
    eigenvalues, V = np.linalg.eigh(ata)
    
    # Sort by eigenvalue magnitude (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    print("2. Eigenvalues of A^T*A:", eigenvalues)
    
    # 3. Singular values (Sigma)
    sigma_vals = np.sqrt(np.maximum(eigenvalues, 0))
    S = np.zeros(A.shape)
    np.fill_diagonal(S, sigma_vals)
    
    # 4. Find U using U_i = (1/sigma_i) * A * V_i
    U = np.zeros((A.shape[0], A.shape[0]))
    for i in range(len(sigma_vals)):
        if sigma_vals[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / sigma_vals[i]
            
    return U, S, V.T

def pca_step_by_step(X, n_components):
    print("\n--- Starting PCA ---")
    # 1. Mean Centering
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    print("1. Data Centered (Subtract Mean)")
    
    # 2. Covariance Matrix
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    print("2. Covariance Matrix:\n", cov_matrix)
    
    # 3. EVD on Covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("3. Explained Variance (Eigenvalues):", eigenvalues)
    
    # 4. Projection
    W = eigenvectors[:, :n_components]
    print(f"4. Projecting onto top {n_components} components")
    return X_centered @ W

# --- Execution ---

A = np.array([[4, 3], [6, 3]], dtype=float)

# Task: Recursive Det
print(f"Final Recursive Det: {recursive_determinant(A.tolist())}\n")

# Task: LU and Det
L, U = lu_decomposition_step_by_step(A)
print
