import numpy as np

def main():
    # np.random.seed(42)
    # lambdas = np.random.uniform(2**(-2), 2**(-3), 3)
    # lambdas = np.append(lambdas, np.random.uniform(2**(-3), 2**(-4), 4))
    # lambdas = np.append(lambdas, np.random.uniform(2**(-4), 2**(-5), 3))
    # lambdas = [float("%0.4E"%l) for l in lambdas]
    # lambdas = np.unique(lambdas)
    lambdas = []
    for i in range(1, 11):
        lambdas.append(2**(-i))
    lambdas = [float("%0.4E"%l) for l in lambdas]
    lambdas = np.unique(lambdas)
    return lambdas
