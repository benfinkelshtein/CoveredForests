import numpy as np

# Appendix Table 3
S = np.array([150, 3288, 22216, 32902])
m_n_d_3 = np.array([139, 3838, 25465, 34794])
M = np.array([5.890, 2.389, 0.497, 1.704])
n = np.array([28, 111, 244, 222])
C_FD = np.array([0.377, 0.202, 0.101, 0.107])
delta = 0.05  # Example value for delta

# Calculate C_tilde
C_tilde = 2 * C_FD / n

# Parameters
log_delta = np.log(1 / delta)  # Compute log(1/delta)


# Define the function to minimize
def our_objective_function(k, m_n_d_3, M, C_tilde):
    term1 = 4 * C_tilde * k
    term2 = M * np.sqrt((m_n_d_3 / (k + 1)) * 4 * np.log(2) + 2 * log_delta) / m_n_d_3
    return term1 + term2


def vc_objective_function(k, m_n_d_3, M, C_tilde):
    term1 = 4 * C_tilde * k
    term2 = M * np.sqrt(m_n_d_3 * 4 * np.log(2) + 2 * log_delta) / m_n_d_3
    return term1 + term2

# Iterate over possible values of k
k_values = range(1, 101)  # Example range for k (1 to 100)

our_optimal_results = []
for i in range(len(m_n_d_3)):
    results = [(k, our_objective_function(k, m_n_d_3[i], M[i], C_tilde[i])) for k in k_values]
    optimal_k, min_value = min(results, key=lambda x: x[1])
    our_optimal_results.append((optimal_k, min_value))

# Display the optimal k and minimal value for each column
for idx, (k, value) in enumerate(our_optimal_results):
    print(f"Column {idx + 1}: Our optimal k = {k}, Minimal Value = {value:.6f}")
print()

vc_optimal_results = []
for i in range(len(m_n_d_3)):
    results = [(k, vc_objective_function(k, m_n_d_3[i], M[i], C_tilde[i])) for k in k_values]
    optimal_k, min_value = min(results, key=lambda x: x[1])
    vc_optimal_results.append((optimal_k, min_value))

# Display the optimal k and minimal value for each column
for idx, (k, value) in enumerate(vc_optimal_results):
    print(f"Column {idx + 1}: VC optimal k = {k}, Minimal Value = {value:.6f}")