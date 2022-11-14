import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats as st
from scipy.stats import sem

def read_lines(filename, datatype, drop_EOS=True):
    lines = []
    with open(filename) as file:
        for line in file:
            l = list(map(datatype, line.rstrip().split()))
            if drop_EOS:
                l = l[:-1]
            lines.append(l)
    return lines

def get_hall(labels, entropies):
    hall = []
    non_hall = []
    labels = sum(labels, [])
    entropies = sum(entropies, [])
    for is_hal, ent in zip(labels, entropies):
        if is_hal:
            hall.append(ent)
        else:
            non_hall.append(ent)
    return hall, non_hall

def get_first_hall(labels, entropies):
    first_hall = []
    for i in range(len(labels)):
        prev_label = 0
        for is_hal, ent in zip(labels[i], entropies[i]):
            is_first = is_hal and not prev_label
            if is_first:
                first_hall.append(ent)
            prev_label = is_hal

    return first_hall

def get_subseq_hall(labels, entropies):
    subseq_hall = []
    for i in range(len(labels)):
        prev_label = 0
        for is_hal, ent in zip(labels[i], entropies[i]):
            is_subseq  = is_hal and prev_label
            if is_subseq:
                subseq_hall.append(ent)
            prev_label = is_hal
    return subseq_hall

def t_test(x, y):
    t_stat, p_val = st.ttest_ind(x, y)
    if p_val <= 0.01:
        res = "Reject"
    else:
        res = "Cannot reject"
    print((res, "%.4E"%t_stat, "%.4E"%p_val))

def ks_test(x, y):
    ks_stat, p_val = st.ks_2samp(x, y)
    if p_val <= 0.01:
        res = "Reject"
    else:
        res = "Cannot reject"
    print((res, "%.4E"%ks_stat, "%.4E"%p_val))

def get_max_val(x, y):
    return np.ceil(max(max(x), max(y)))

def distrib(x, y, saveplots, showplots, filename=None, x_name="Hallucinated", y_name="Non-Hallucinated"):
    max_val = get_max_val(x, y)
    bins = np.linspace(0, max_val, 100)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2


    fig = plt.figure(figsize=(10, 6), dpi=100)

    plt.hist(x, alpha=0.5, bins=bins, density=True, label=x_name)
    plt.hist(y, alpha=0.5, bins=bins, density=True, label=y_name)
    plt.xlabel('Entropy', labelpad=10)
    plt.ylabel('Frequency', labelpad=10)
    plt.legend(frameon=False, loc='upper right', fontsize=16)
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if saveplots:
        path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/tmp/"
        plt.savefig(path + filename)

    if showplots:
        plt.show()

def display_statistics(labels, hall, non_hall, first_hall, subseq_hall):
    print("")
    print("Average Entropies, Standard error")
    print("Hallucinated: %.4f, %.4f" % (np.mean(hall), sem(hall)))
    print("Non-Hallucinated: %.4f, %.4f" % (np.mean(non_hall), sem(non_hall)))
    print("Initial Hallucinated: %.4f, %.4f" % (np.mean(first_hall), sem(first_hall)))
    print("Subsequent Hallucinated: %.4f, %.4f" % (np.mean(subseq_hall), sem(subseq_hall)))

    print("")
    print("Non-Hallucinated vs Hallucinated, Initial Hallucinated, Subsequent Hallucinated")

    print("")
    print("T-test")
    t_test(hall, non_hall)
    t_test(first_hall, non_hall)
    t_test(subseq_hall, non_hall)

    print("")
    print("KS-test")
    ks_test(hall, non_hall)
    ks_test(first_hall, non_hall)
    ks_test(subseq_hall, non_hall)

    print("")
    print("Initial vs Subsequent Hallucinated")

    print("")
    print("T-test")
    t_test(subseq_hall, first_hall)

    print("")
    print("KS-test")
    ks_test(subseq_hall, first_hall)

def handle_plots(hall, non_hall, first_hall, subseq_hall, showplots, saveplots, plot_prefix):
    distrib(hall, non_hall, saveplots, showplots, filename=plot_prefix + "ent_hall_vs_non_hall")
    distrib(first_hall, non_hall, saveplots, showplots, filename=plot_prefix + "ent_first_hall_vs_non_hall", x_name="First Hallucinated")
    distrib(subseq_hall, non_hall, saveplots, showplots, filename=plot_prefix + "ent_subseq_hall_vs_non_hall", x_name="Subsequent Hallucinated")
    distrib(first_hall, subseq_hall, saveplots, showplots, filename=plot_prefix + "ent_first_hall_vs_subseq_hall", x_name="First Hallucinated", y_name="Subsequent Hallucinated")

def display_results(labels, entropies, showplots=True, saveplots=False, plot_prefix=""):
    # Sanity check
    assert len(labels) == len(entropies)
    for i, j in zip(labels, entropies):
        if not (len(i) == len(j)):
            print(len(i), " ".join(map(str, i)))
            print(len(j), " ".join(map(str, j)))
            assert False
    assert all(len(i) == len(j) for i, j in zip(labels, entropies))

    hall, non_hall = get_hall(labels, entropies)
    first_hall = get_first_hall(labels, entropies)
    subseq_hall = get_subseq_hall(labels, entropies)

    display_statistics(labels, hall, non_hall, first_hall, subseq_hall)
    handle_plots(hall, non_hall, first_hall, subseq_hall, showplots, saveplots, plot_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute entropy statistics')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    base_path = "data/xsum-hallucination"
    if args.dataset == "bart":
        base_path += "-bart"
    base_path += "/"

    label_filename = base_path + "test.label"
    labels = read_lines(label_filename, int, drop_EOS=False)

    entropy_filename = base_path + "test.entropy.sm"
    entropies = read_lines(entropy_filename, float)
    display_results(labels, entropies, showplots=False, saveplots=True, plot_prefix=args.dataset + "_sm_")

    entropy_filename = base_path + "test.entropy.lm"
    entropies = read_lines(entropy_filename, float)
    display_results(labels, entropies, showplots=False, saveplots=True, plot_prefix=args.dataset + "_lm_")
