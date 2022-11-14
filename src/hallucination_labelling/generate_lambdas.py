import numpy as np

def main():
    np.random.seed(42)
    lambdas = np.random.uniform(0.05, 0.15, 10)
    # lambdas = []
    # for i in range(1, 11):
    #     lambdas.append(2**(-i))
    lambdas = [float("%0.4E"%l) for l in lambdas]
    lambdas = np.unique(lambdas)
    return lambdas
