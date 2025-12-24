import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def solve_ode_ultimate_and_plot(coeffs, initial_conditions=None, f_type=None, f_val=None, x_range=(0, 10)):
    """
    Solves a_n y^(n) + ... + a_0 y = f(x)
    coeffs: [a_n, ..., a_0]
    initial_conditions: [y(0), y'(0), ...]
    f_type: 'exp', 'trig'
    f_val: 'a' for e^{ax} or 'b' for cos(bx)/sin(bx)
    """
    # --- 1. Characteristic Equation & Roots ---
    roots = np.roots(coeffs)
    rounded_roots = [complex(round(r.real, 4), round(r.imag, 4)) for r in roots]
    root_counts = Counter(rounded_roots)
    
    basis_funcs = []
    basis_strs = []
    processed = set()
    sorted_roots = sorted(root_counts.items(), key=lambda x: (abs(x[0].imag), x[0].real))

    # --- 2. Construct Basis (Homogeneous) ---
    for root, mult in sorted_roots:
        if root in processed: continue
        if abs(root.imag) < 1e-4:  # Real Root
            for m in range(mult):
                basis_funcs.append(lambda x, r=root.real, m=m: (x**m) * np.exp(r*x))
                x_s = "" if m == 0 else ("x" if m == 1 else f"x^{m}")
                e_s = f"e^({root.real}x)" if root.real != 0 else ("1" if m == 0 else "")
                basis_strs.append(f"{x_s}{e_s}".replace("1e", "e").strip("1"))
        else:  # Complex Root
            alpha, beta = root.real, abs(root.imag)
            processed.update([root, complex(alpha, -beta)])
            for m in range(mult):
                basis_funcs.append(lambda x, a=alpha, b=beta, m=m: (x**m) * np.exp(a*x) * np.cos(b*x))
                basis_funcs.append(lambda x, a=alpha, b=beta, m=m: (x**m) * np.exp(a*x) * np.sin(b*x))
                x_s = "" if m == 0 else ("x" if m == 1 else f"x^{m}")
                e_s = f"e^({alpha}x)" if abs(alpha) > 1e-4 else ""
                basis_strs.append(f"{x_s}{e_s}cos({beta}x)")
                basis_strs.append(f"{x_s}{e_s}sin({beta}x)")

    # --- 3. Initial Value Problem Solver ---
    # We solve Ay = b where A is the Wronskian at x=0
    c_vals = None
    if initial_conditions:
        n = len(coeffs) - 1
        A_mat = np.zeros((n, n))
        h = 1e-5
        for j in range(n):
            for i in range(n):
                if i == 0: A_mat[i, j] = basis_funcs[j](0)
                elif i == 1: A_mat[i, j] = (basis_funcs[j](h) - basis_funcs[j](-h)) / (2*h)
                else: A_mat[i, j] = (basis_funcs[j](h) - 2*basis_funcs[j](0) + basis_funcs[j](-h)) / (h**2)
        
        try:
            c_vals = np.linalg.solve(A_mat, initial_conditions)
        except np.linalg.LinAlgError:
            print("Warning: Could not solve for constants.")

    # --- 4. Define the Final Function for Plotting ---
    def y_final(x):
        if c_vals is not None:
            return sum(c * f(x) for c, f in zip(c_vals, basis_funcs))
        return sum(f(x) for f in basis_funcs) # Default C=1 if not specified

    # --- 5. Formatting Output String ---
    c_labels = [str(round(c, 3)) for c in c_vals] if c_vals is not None else [f"C_{i+1}" for i in range(len(basis_strs))]
    yh_str = " + ".join([f"{c}{s}" for c, s in zip(c_labels, basis_strs)])
    
    # Non-Homogeneous Labeling
    yp_str = ""
    if f_type == 'exp':
        res = root_counts.get(complex(f_val, 0), 0)
        yp_str = f" + A{'x' if res > 0 else ''}e^({f_val}x)"
    elif f_type == 'trig':
        res = root_counts.get(complex(0, f_val), 0)
        yp_str = f" + {'x' if res > 0 else ''}(A cos({f_val}x) + B sin({f_val}x))"

    print(f"General Solution: y(x) = {yh_str}{yp_str}")

    # --- 6. Plotting ---
    x_vec = np.linspace(x_range[0], x_range[1], 500)
    y_vec = [y_final(val) for val in x_vec]

    plt.figure(figsize=(10, 5))
    plt.plot(x_vec, y_vec, label="y(x)", color='blue', linewidth=2)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title(f"ODE Solution Plot: {coeffs}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# --- RUNNING THE TEST CASES ---

# Case A: Damped Vibration (y'' + 0.5y' + 2y = 0)
print("--- Case A: Damped Oscillation ---")
solve_ode_ultimate_and_plot([1, 0.5, 2], [1, 0])

# Case B: Resonance (y'' + 4y = 0 with force freq matching natural freq)
print("\n--- Case B: Identifying Resonance ---")
solve_ode_ultimate_and_plot([1, 0, 4], f_type='trig', f_val=2)
