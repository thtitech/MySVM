from sklearn.metrics.pairwise import *

kernel_dct = {
    "rbf": rbf_kernel,
    "linear": linear_kernel
}

eps = 10**(-9)
