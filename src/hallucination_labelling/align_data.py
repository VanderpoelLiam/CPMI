import argparse

def read_lines(filename, start):
    lines = []
    order = []
    with open(filename) as file:
        for line in file:
            if start in line:
                l = list(map(str, line.rstrip().split()))
                clean_line = l[1:]
                clean_order = int(l[0].split(start)[1])
                lines.append(clean_line)
                order.append(clean_order)
    lines = [x for _, x in sorted(zip(order, lines))]
    return lines

def write_lines(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            f.write("%s\n" % " ".join(d))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align the data with the hallucination labels')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    filename = "logs/experiments/token_level_entropy_" + args.dataset
    out_path = "data/xsum-hallucination"
    if args.dataset == "bart":
        out_path += "-bart"
    out_path += "/"

    ents = read_lines(filename, "ENT_SM-")
    lang_ents = read_lines(filename, "ENT_LM-")
    sm = read_lines(filename, "P_SM-")
    lm = read_lines(filename, "P_LM-")

    write_lines(out_path + "test.entropy.sm", ents)
    write_lines(out_path + "test.entropy.lm", lang_ents)
    write_lines(out_path + "test.prob.sm", sm)
    write_lines(out_path + "test.prob.lm", lm)
