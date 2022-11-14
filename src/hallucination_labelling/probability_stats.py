import numpy as np
import argparse
from entropy_stats import read_lines, get_hall, get_first_hall, get_subseq_hall
from scipy.stats import sem

def display_statistics(labels, hall, non_hall, first_hall, subseq_hall):
    print("")
    print("Average Probabilities, Standard Error")
    print("Hallucinated:            %.4f, %.4f" % (np.mean(hall), sem(hall)))
    print("Non-Hallucinated:        %.4f, %.4f" % (np.mean(non_hall), sem(non_hall)))
    print("Initial Hallucinated:    %.4f, %.4f" % (np.mean(first_hall), sem(first_hall)))
    print("Subsequent Hallucinated: %.4f, %.4f" % (np.mean(subseq_hall), sem(subseq_hall)))

def display_results(labels, probs):
    # Sanity check
    assert len(labels) == len(probs)
    assert all(len(i) == len(j) for i, j in zip(labels, probs))

    hall, non_hall = get_hall(labels, probs)
    first_hall = get_first_hall(labels, probs)
    subseq_hall = get_subseq_hall(labels, probs)

    display_statistics(labels, hall, non_hall, first_hall, subseq_hall)


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

    sm_filename = base_path + "test.prob.sm"
    lm_filename = base_path + "test.prob.lm"
    probs_sm = read_lines(sm_filename, float)
    probs_lm = read_lines(lm_filename, float)

    print("\nSUMMARIZATION MODEL")
    display_results(labels, probs_sm)
    print("\nLANGUAGE MODEL")
    display_results(labels, probs_lm)
