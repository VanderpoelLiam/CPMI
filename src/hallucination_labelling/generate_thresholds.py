import numpy as np

def main():
    """
    * Idea 1: is that ratio of the PDFs of the distributions at the point
    x tells us the probability of belonging to one distribution over another.
    We can do gridsearch over x values in range [0.5, 1] probability of
    belonging to the first hallucination distribution. Use this (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html)
    to get the pdfs.

    * Idea 2: Pick area around center of mass between first hallucinated vs
    non-hallucinated distributions as the trigger value for MMI decoding
    """
    np.random.seed(42)
    thresholds = np.random.uniform(3, 4, 10)
    # Simple grid search over area around the mean.
    # np.random.seed(42)
    # mean = 3.1147
    # std = 1.3637
    # thresholds = np.random.uniform(mean - std, mean + std, 5)
    thresholds = [float("%0.4E"%t) for t in thresholds]
    thresholds = np.unique(thresholds)
    return thresholds
