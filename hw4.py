import numpy as np

def poly_roots(coeffs):
    while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-14:
        coeffs.pop()

    deg = len(coeffs) - 1

    if deg == 0:
        return "no roots"
    if deg == 1:
        return [-coeffs[0] / coeffs[1]]

    C = np.zeros((deg, deg))
    C[1:, :-1] = np.eye(deg - 1)
    C[0, :] = -np.array(coeffs[:-1])

    roots = np.linalg.eigvals(C)
    return roots

poly = [-8, 14, -7, 1]
print(poly_roots(poly))
