import math
import numpy as np

# probability
def calculate_probabilities():
    prob_direct = 0.5 ** 10000
    log_val = 10000 * math.log(0.5)
    return prob_direct, log_val

def entropy(p):
    return -np.sum(p * np.log2(p + 1e-10))

def cross_entropy(p, q):
    return -np.sum(p * np.log2(q + 1e-10))

def kl_divergence(p, q):
    return np.sum(p * np.log2((p / (q + 1e-10)) + 1e-10))

def mutual_information(px, py, pxy):
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return mi

def hamming_encode(data_4bit):
    G = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(G, data_4bit) % 2

def hamming_decode(received_7bit):
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    syndrome = np.dot(H, received_7bit) % 2
    syndrome_int = int("".join(syndrome.astype(str)[::-1]), 2)
    
    if syndrome_int != 0:
        error_idx = syndrome_int - 1
        received_7bit[error_idx] = (received_7bit[error_idx] + 1) % 2
    
    return received_7bit[[2, 4, 5, 6]]

if __name__ == "__main__":
    prob, log_prob = calculate_probabilities()
    print(f"Direct 0.5^10000: {prob}")
    print(f"Log calculation: {log_prob}")
    print("-" * 20)

    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.1, 0.7, 0.2])
    
    print(f"Entropy(p): {entropy(p)}")
    print(f"Cross Entropy(p,q): {cross_entropy(p, q)}")
    print(f"KL Divergence(p||q): {kl_divergence(p, q)}")
    
    pxy = np.array([[0.1, 0.1], [0.0, 0.8]])
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    print(f"Mutual Info: {mutual_information(px, py, pxy)}")
    print("-" * 20)

    h_pq = cross_entropy(p, q)
    h_pp = cross_entropy(p, p)
    print(f"H(p,q): {h_pq}")
    print(f"H(p,p): {h_pp}")
    print(f"H(p,q) > H(p,p): {h_pq > h_pp}")
    print("-" * 20)

    original_data = np.array([1, 0, 1, 1])
    encoded_data = hamming_encode(original_data)
    print(f"Encoded: {encoded_data}")

    corrupted_data = encoded_data.copy()
    corrupted_data[2] = (corrupted_data[2] + 1) % 2
    print(f"Corrupted: {corrupted_data}")

    decoded_data = hamming_decode(corrupted_data)
    print(f"Decoded: {decoded_data}")