import cmath

def root2(a, b, c):
    D = b*b - 4*a*c
    r1 = (-b + cmath.sqrt(D)) / (2*a)
    r2 = (-b - cmath.sqrt(D)) / (2*a)
    return (r1, r2)

print(root2(1, -5, 6))
print(root2(1, 4, 3))
print(root2(1, 1, 1))
