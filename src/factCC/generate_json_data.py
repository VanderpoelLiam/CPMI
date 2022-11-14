import json

def write_lines(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")


def read_lines(filename):
    lines = []
    with open(filename) as file:
        for line in file:
            lines.append(line.rstrip())
    return lines

def main():
    base_path = "data/factCC/"
    suffixes = ["bart_0.0000E+00_0.0000E+00",
        "bart_6.5602E-02_3.5987E+00",
        "standard_0.0000E+00_0.0000E+00",
        "standard_1.3120E-01_3.5618E+00"]

    for suffix in suffixes:
        data = []
        hypo_filename = base_path + "hypo_" + suffix
        source_filename = base_path + "source_" + suffix
        json_filename = base_path + "data-dev_" + suffix + ".jsonl"
        hypos = read_lines(hypo_filename)
        sources = read_lines(source_filename)
        for i, (hypo, source) in enumerate(zip(hypos, sources)):
            datum = {"claim": hypo, "text": source, "id": str(i), "label": "CORRECT"}
            data.append(datum)
        write_lines(json_filename, data)

if __name__ == '__main__':
    main()
