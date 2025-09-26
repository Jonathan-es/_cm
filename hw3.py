import cmath

def cubic_roots(a, b, c, d):
    B, C, D = b/a, c/a, d/a

    P = C - (B*B)/3
    Q = (2*B*B*B)/27 - (B*C)/3 + D

    Delta = (Q/2)**2 + (P/3)**3

    U = cmath.sqrt(Delta)
    Uc = (-Q/2 + U) ** (1/3)
    Vc = (-Q/2 - U) ** (1/3)

    w = [
        1,
        -0.5 + cmath.sqrt(3)/2*1j,
        -0.5 - cmath.sqrt(3)/2*1j
    ]

    result = []
    for k in range(3):
        Y = Uc*w[k] + Vc*w[(3-k) % 3]
        X = Y - B/3
        result.append(X)

    return result

print(cubic_roots(1, -6, 11, -6))
