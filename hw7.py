import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest

np.random.seed(42)

def print_separator(title):
    print(f"\n--- {title} ---")

# 1. One-Sample Z-Test
# Scenario: Testing if sample mean differs from population mean (100)
# Assumption: Population standard deviation is KNOWN (sigma = 15)
print_separator("1. One-Sample Z-Test")

mu_null = 100
sigma_pop = 15
n = 50
data_z = np.random.normal(loc=105, scale=sigma_pop, size=n)

# Manual Calculation
x_bar = np.mean(data_z)
z_score_manual = (x_bar - mu_null) / (sigma_pop / np.sqrt(n))
p_value_manual = 2 * (1 - stats.norm.cdf(abs(z_score_manual))) # Two-tailed

# Library Calculation
z_score_lib, p_value_lib = ztest(data_z, value=mu_null)

print(f"Sample Mean: {x_bar:.4f}")
print(f"Manual Z: {z_score_manual:.4f}, P-value: {p_value_manual:.4f}")
print(f"Library Z: {z_score_lib:.4f}, P-value: {p_value_lib:.4f}")


# 2. One-Sample T-Test
# Scenario: Testing if sample mean differs from population mean (50)
# Assumption: Population std dev is UNKNOWN
print_separator("2. One-Sample T-Test")

mu_null_t = 50
n_t = 20
data_t = np.random.normal(loc=48, scale=5, size=n_t)

# Manual Calculation
x_bar_t = np.mean(data_t)
s_sample = np.std(data_t, ddof=1) # ddof=1 for sample std dev
t_stat_manual = (x_bar_t - mu_null_t) / (s_sample / np.sqrt(n_t))
df = n_t - 1
p_val_t_manual = 2 * (1 - stats.t.cdf(abs(t_stat_manual), df))

# Library Calculation
t_stat_lib, p_val_t_lib = stats.ttest_1samp(data_t, mu_null_t)

print(f"Sample Mean: {x_bar_t:.4f}")
print(f"Manual T: {t_stat_manual:.4f}, P-value: {p_val_t_manual:.4f}")
print(f"Library T: {t_stat_lib:.4f}, P-value: {p_val_t_lib:.4f}")


# 3. Independent Two-Sample T-Test
# Scenario: Comparing Group A vs Group B
# Assumption: Equal variances (pooled)
print_separator("3. Independent Two-Sample T-Test")

n1, n2 = 30, 30
group_a = np.random.normal(loc=10, scale=2, size=n1)
group_b = np.random.normal(loc=12, scale=2, size=n2)

# Manual Calculation
mean1, mean2 = np.mean(group_a), np.mean(group_b)
var1, var2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

# Pooled Variance
sp_squared = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
sp = np.sqrt(sp_squared)
se_diff = sp * np.sqrt(1/n1 + 1/n2)

t_ind_manual = (mean1 - mean2) / se_diff
df_ind = n1 + n2 - 2
p_ind_manual = 2 * (1 - stats.t.cdf(abs(t_ind_manual), df_ind))

# Library Calculation
t_ind_lib, p_ind_lib = stats.ttest_ind(group_a, group_b, equal_var=True)

print(f"Mean A: {mean1:.4f}, Mean B: {mean2:.4f}")
print(f"Manual T: {t_ind_manual:.4f}, P-value: {p_ind_manual:.4f}")
print(f"Library T: {t_ind_lib:.4f}, P-value: {p_ind_lib:.4f}")


# 4. Paired Two-Sample T-Test
# Scenario: Before vs After (same subjects)
print_separator("4. Paired Two-Sample T-Test")

n_pair = 25
before = np.random.normal(loc=60, scale=10, size=n_pair)
noise = np.random.normal(loc=-5, scale=3, size=n_pair) # Represents treatment effect
after = before + noise

# Manual Calculation (Reduce to One-Sample on differences)
differences = before - after
d_bar = np.mean(differences)
s_d = np.std(differences, ddof=1)
t_pair_manual = d_bar / (s_d / np.sqrt(n_pair))
df_pair = n_pair - 1
p_pair_manual = 2 * (1 - stats.t.cdf(abs(t_pair_manual), df_pair))

# Library Calculation
t_pair_lib, p_pair_lib = stats.ttest_rel(before, after)

print(f"Mean Difference: {d_bar:.4f}")
print(f"Manual T: {t_pair_manual:.4f}, P-value: {p_pair_manual:.4f}")
print(f"Library T: {t_pair_lib:.4f}, P-value: {p_pair_lib:.4f}")