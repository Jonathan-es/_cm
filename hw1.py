h = 0.00001

def df(f, x):
    return (f(x + h) - f(x)) / h

def integral(f, a, b):
    x, area = a, 0
    while x < b:
        area += f(x) * h
        x += h
    return area

def theorem1(f, x):
    r = df(lambda t: integral(f, 0, t), x)
    print("r =", r, "f(x) =", f(x))
    print("abs(r - f(x)) < 0.01 =", abs(r - f(x)) < 0.01)
    assert abs(r - f(x)) < 0.01

def f(x):
    return x**3

print("df(f, 2) =", df(f, 2))
print("integral(f, 0, 2) =", integral(f, 0, 2))

theorem1(f, 2)
