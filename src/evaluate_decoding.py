import argparse
import numpy as np
import subprocess
import extract_score
import torch
from hallucination_labelling.entropy_stats import get_hall, get_first_hall, get_subseq_hall
from scipy.stats import sem
from bert_score import score

def extract_lines(filename, start):
    with open(filename, "r") as f:
        lines = []
        for line in f:
            if line.startswith(start):
                lines.append(line)
    return lines

def read_logs(filename, start, type, drop_last=True):
    """
    Last value for probability and entropy sequences is the EOS token value.
    The hypothesis sequence is therefore one value shorter than these
    sequences. Hence we need to drop the last token value in these cases.
    """
    lines = extract_lines(filename, start)
    values = []
    for line in lines:
        l = list(map(str, line.rstrip().split()))
        clean_line = list(map(type, l[1:]))
        if drop_last:
            del clean_line[-1]
        values.append(clean_line)
    return values

def read_labels(filename):
    with open(filename, "r") as f:
        lines = []
        for line in f:
            if line.startswith("FAIL"):
                lines.append(None)
            else:
                lines.append(list(map(int, line.rstrip().split())))
    return lines

def display_statistics(type, labels, hall, non_hall, first_hall, subseq_hall, isRanking=False):
    print(type + " STATISTICS")
    print("Average, Standard Error")
    if isRanking:
        print("Hallucinated:            %d \pm %d" % (round(np.mean(hall)), round(sem(hall))))
        print("Non-Hallucinated:        %d \pm %d" % (round(np.mean(non_hall)), round(sem(non_hall))))
        print("Initial Hallucinated:    %d \pm %d" % (round(np.mean(first_hall)), round(sem(first_hall))))
        print("Subsequent Hallucinated: %d \pm %d" % (round(np.mean(subseq_hall)), round(sem(subseq_hall))))
    else:
        print("Hallucinated:            %.2f \pm %.2f" % (np.mean(hall), sem(hall)))
        print("Non-Hallucinated:        %.2f \pm %.2f" % (np.mean(non_hall), sem(non_hall)))
        print("Initial Hallucinated:    %.2f \pm %.2f" % (np.mean(first_hall), sem(first_hall)))
        print("Subsequent Hallucinated: %.2f \pm %.2f" % (np.mean(subseq_hall), sem(subseq_hall)))
    print("")

def display_results(type, labels, values, isRanking=False):
    # Sanity checks
    assert len(labels) == len(values)
    for i, j in zip(labels, values):
        if not (len(i) == len(j)):
            print(len(i), i)
            print(len(j), j)
            assert False
    assert all(len(i) == len(j) for i, j in zip(labels, values))

    hall, non_hall = get_hall(labels, values)
    first_hall = get_first_hall(labels, values)
    subseq_hall = get_subseq_hall(labels, values)

    display_statistics(type, labels, hall, non_hall, first_hall, subseq_hall, isRanking)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate decoding for the given parameters')

    parser.add_argument('lamb', type=float,
                        help='Lambda value')

    parser.add_argument('threshold', type=float,
                        help='Entropy threshold value')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    label_dir = "data/xsum-detect-hall/"
    log_dir = "logs/experiments/"

    suffix = "%0.4E_%0.4E" % (args.lamb, args.threshold)
    label_filename = label_dir + "label_processed_" + args.dataset + "_ref"
    log_filename = log_dir + args.dataset + "_ref_no_500_" + suffix

    labels = read_labels(label_filename)
    probs = read_logs(log_filename, "P-", float)
    ranks = read_logs(log_filename, "RANK-", int)

    missing_inds = [i for i,v in enumerate(labels) if v == None]
    for i in sorted(missing_inds, reverse=True):
        del labels[i]
        del probs[i]
        del ranks[i]

    display_results("PROBABILITY", labels, probs)
    display_results("RANKING", labels, ranks, True)

    rouge_dir = "logs/rouge/"
    no_500_filename = args.dataset + "_no_500_" + suffix
    rouge_dir += no_500_filename + "/"
    rouge_score_filename = rouge_dir + "score"
    try:
        f = open(rouge_score_filename, "r")
        f.close()
        extract_score.extract(rouge_score_filename)
    except FileNotFoundError:
        compute_rouge = "src/score_generate.sh " + no_500_filename
        if args.dataset == "bart":
            compute_rouge = "src/score_generate.sh -b " + no_500_filename
        subprocess.call(compute_rouge, shell=True)

    with open(rouge_dir + "hypothesis.detok") as f:
        cands = [line.strip() for line in f]

    with open(rouge_dir + "target.detok") as f:
        refs = [line.strip() for line in f]

    try:
        P = torch.load(rouge_dir + 'bert_score_p.pt')
        R = torch.load(rouge_dir + 'bert_score_r.pt')
        F1 = torch.load(rouge_dir + 'bert_score_f1.pt')
    except FileNotFoundError:
        P, R, F1 = score(cands, refs, lang='en', verbose=False)
        torch.save(P, rouge_dir + 'bert_score_p.pt')
        torch.save(R, rouge_dir + 'bert_score_r.pt')
        torch.save(F1, rouge_dir + 'bert_score_f1.pt')


    print(f"\nBERTS P: {P.mean():.3f}")
    print(f"BERTS R: {R.mean():.3f}")
    print(f"BERTS F1: {F1.mean():.3f}")

    label_filename = label_dir + "label_processed_" + "_".join([args.dataset, suffix])
    labels = read_labels(label_filename)

    missing_inds = [i for i,v in enumerate(labels) if v == None]
    for i in sorted(missing_inds, reverse=True):
        del labels[i]
    labels_flat = [x for xs in labels for x in xs]

    print(f"\nFactScore: {1 -np.mean(labels_flat):.3f}")
