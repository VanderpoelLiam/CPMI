# Align the Gold.target labels with labels for sentences after tokenisation and bpe
import simalign
import textwrap
import argparse

def get_out(words, longest):
    out = ""
    for w in words:
        spaces = longest - len(w)
        out += w + spaces*" "
    return out

def make_pretty(sent):
    wrapped = textwrap.fill(sent, 70)
    return wrapped.split("\n")

def get_issue_inds(refs, trgs, trg_labels):
    inds = []
    for i in range(500):
        trg_label_split = trg_labels[i].split()
        ref_split = refs[i].split()
        trg_split = trgs[i].split()

        ref_len = len(ref_split)
        target_len = len(trg_split)

        if (ref_len > target_len):
            pass
        elif (ref_len < target_len):
            inds.append(i)
        else:
            pass
    return inds

def print_pretty(i, refs, trgs, trg_labels, longest=15):
    trg_label_split = trg_labels[i].split()
    ref_split = refs[i].split()
    trg_split = trgs[i].split()
    indices = map(str, list(range(len(trg_split))))

    trg_out = get_out(trg_split, longest)
    trg_label_out = get_out(trg_label_split, longest)
    ref_out = get_out(ref_split, longest)
    indices_out = get_out(indices, longest)

    trg_pretty = make_pretty(trg_out)
    trg_label_pretty = make_pretty(trg_label_out)
    ref_pretty = make_pretty(ref_out)
    indices_pretty = make_pretty(indices_out)

    max_len = max(len(indices_pretty), len(trg_pretty), len(trg_label_pretty), len(ref_pretty))

    while len(indices_pretty) < max_len:
        indices_pretty.append("")
    while len(trg_pretty) < max_len:
        trg_pretty.append("")
    while len(trg_label_pretty) < max_len:
        trg_label_pretty.append("")
    while len(ref_pretty) < max_len:
        ref_pretty.append("")

    print("index: ", i+1)
    for j, t, l, r in zip(indices_pretty, trg_pretty, trg_label_pretty, ref_pretty):
        print(j)
        print(t)
        print(l)
        # print(r)
        print("\n")
    print(trg_labels[i])

def print_ref_and_label(i, ref, ref_label, longest=15):
    ref_label_split = ref_label.split()
    ref_split = ref.split()
    indices = map(str, list(range(len(ref_split))))

    ref_label_out = get_out(ref_label_split, longest)
    ref_out = get_out(ref_split, longest)
    indices_out = get_out(indices, longest)

    ref_label_pretty = make_pretty(ref_label_out)
    ref_pretty = make_pretty(ref_out)
    indices_pretty = make_pretty(indices_out)

    max_len = max(len(indices_pretty), len(ref_label_pretty), len(ref_pretty))

    while len(indices_pretty) < max_len:
        indices_pretty.append("")
    while len(ref_label_pretty) < max_len:
        ref_label_pretty.append("")
    while len(ref_pretty) < max_len:
        ref_pretty.append("")

    print("index: ", i+1)
    for j, r, l in zip(indices_pretty, ref_pretty, ref_label_pretty):
        print(j)
        print(r)
        print(l)
        print("\n")
    print(ref_label)

def get_data(data_dir):
    ref_path = "data/xsum-hallucination-raw/"
    ref = ref_path +"Gold.target"
    ref_label = ref_path + "Gold.label"

    refs = []
    with open(ref) as f:
        for line in f:
            refs.append(line.rstrip())

    ref_labels = []
    with open(ref_label) as f:
        for line in f:
            ref_labels.append(line.rstrip())

    target = data_dir + "test.bpe.target"

    targets = []
    with open(target) as f:
        for line in f:
            targets.append(line.rstrip())
    return refs, ref_labels, targets

def get_target_label(ref_label, aligns, len_target):
    ref_label_split = ref_label.split()
    target_label_split = []
    for i in range(len_target):
        target_label_split.append("?")

    itermax_aligns = aligns['itermax']
    match_aligns = aligns['mwmf']
    for (i, j) in itermax_aligns:
        target_label_split[j] = ref_label_split[i]

    for (i, j) in match_aligns:
        if target_label_split[j] == "?":
            target_label_split[j] = ref_label_split[i]

    target_label = " ".join(target_label_split)
    return target_label

def all_same_val(sent, val):
    sent = sent.replace("?", "")
    sent = sent.replace(" ", "")
    expected = len(sent) * val
    res = (sent == expected)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align labels for hallucination dataset')

    parser.add_argument('--manual', action='store_true',
                        help='Run the manual alignment, otherwise automatically align.')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    data_dir = "data/xsum-hallucination"
    if args.dataset == "bart":
        data_dir += "-bart"
    data_dir += "/"

    refs, ref_labels, targets = get_data(data_dir)


    if args.manual:
        count = 0
        with open(data_dir + "test.label") as f:
            for line in f:
                if ("?" in line) or ("1 0 1" in line):
                    count += 1
        print("# Labels to manually align: ", count)

        # Manually align remaining labels
        with open(data_dir + "test.label") as f:
            for i, target_label in enumerate(f):
                len_target = len(targets[i].split())
                len_target_label = len(target_label.split())
                if (not len_target == len_target_label):
                # if ("?" in target_label) or \
                #    ("1 0 1" in target_label) or \
                #    (not len_target == len_target_label):
                    target_label = target_label.rstrip()
                    print_ref_and_label(i, refs[i], ref_labels[i])
                    print_ref_and_label(i, targets[i], target_label)
                    print("--------------------------------------------")
                    input("Press Enter to continue...")
    else:
        # Automatically align as best we can
        aligner = simalign.SentenceAligner(token_type="bpe")
        with open(data_dir + "test.label", 'w') as fp:
            for i in range(500):
                if i % 10 == 0:
                    print(i)
                len_target = len(targets[i].split())
                aligns = aligner.get_word_aligns(refs[i], targets[i])
                target_label = get_target_label(ref_labels[i], aligns, len_target)

                # Fix obvious errors
                if "?" in target_label:
                    if all_same_val(target_label, "1"):
                        target_label = target_label.replace('?', '1')
                    elif all_same_val(target_label, "0"):
                        target_label = target_label.replace('?', '0')

                fp.write(target_label + '\n')
    print("Done.")
