import argparse
from entropy_stats import read_lines, get_first_hall
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from statistics import mean

def display_results(labels, tokens, probs, showplots=True, saveplots=False, plot_prefix=""):
    # Sanity checks
    assert len(labels) == len(tokens)
    assert len(labels) == len(probs)
    assert all(len(i) == len(j) for i, j in zip(labels, tokens))
    assert all(len(i) == len(j) for i, j in zip(labels, probs))

    tokens_filtered = get_first_hall(labels, tokens)
    probs_filtered = get_first_hall(labels, probs)
    token_probs = defaultdict(list)
    for k, v in zip(tokens_filtered, probs_filtered):
        token_probs[k].append(v)

    token_avg_probs = {}
    for k in tokens_filtered:
        token_avg_probs[k] = mean(token_probs[k])

    token_avg_probs = dict(sorted(token_avg_probs.items(), key=lambda item: item[1], reverse=True))
    num_entries = 20
    probs = list(token_avg_probs.values())[:num_entries]
    labels = list(token_avg_probs.keys())[:num_entries]
    ticks = np.arange(len(token_avg_probs))[:num_entries]

    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.bar(ticks, probs, align='center')
    fig.autofmt_xdate()
    plt.xticks(ticks, labels)

    if saveplots:
        filename = plot_prefix + "first_hallucinated_token_ranking"
        path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/tmp/"
        plt.savefig(path + filename)

    if showplots:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute token statistics for initial hallucinated labelled tokens')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    base_path = "data/xsum-hallucination"
    if args.dataset == "bart":
        base_path += "-bart"
    base_path += "/"

    label_filename = base_path + "test.label"
    tokens_filename = base_path + "test.bpe.target"
    sm_filename = base_path + "test.prob.sm"
    lm_filename = base_path + "test.prob.lm"

    labels = read_lines(label_filename, int, drop_EOS=False)
    tokens = read_lines(tokens_filename, str, drop_EOS=False)
    probs_sm = read_lines(sm_filename, float)
    probs_lm = read_lines(lm_filename, float)

    display_results(labels, tokens, probs_sm, showplots=False, saveplots=True, plot_prefix=args.dataset + "_sm_")
    display_results(labels, tokens, probs_lm, showplots=False, saveplots=True, plot_prefix=args.dataset + "_lm_")
