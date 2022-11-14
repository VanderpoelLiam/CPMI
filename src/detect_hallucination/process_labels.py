import subprocess
import sys
import os
import time
from sacremoses import MosesTokenizer
from fairseq.data.encoders.gpt2_bpe import get_encoder

def encode(bpe, line):
        ids = bpe.encode(line)
        return list(map(str, ids))

def read_lines(filename, datatype):
    lines = []
    with open(filename) as file:
        for line in file:
            l = list(map(datatype, line.rstrip().split()))
            lines.append(l)
    return lines

def process_raw(raw, mt, bpe):
    from_raw = []
    for i, token in enumerate(raw):
        if mt is None:
            if i > 0:
                token = " " + token
            enc_tokens = encode(bpe, token)
            from_raw.append(" ".join(enc_tokens))
        else:
            token_tok = mt.tokenize(token, return_str=True)
            result = subprocess.run(['src/detect_hallucination/apply_bpe.sh', token_tok], capture_output=True, text=True)
            from_raw.append(result.stdout.rstrip())
    return from_raw

def extend_labels_from_raw(raw, from_raw, labels_raw):
    labels_processed = []
    for l_raw, tok_raw, tok_from_raw in zip(labels_raw, raw, from_raw):
        l_processed = [l_raw] * len(tok_from_raw.split())
        labels_processed.extend(l_processed)

    return list(map(str, labels_processed))

def test_validity(processed, from_raw):
    sent_processed = " ".join(processed)
    sent_from_raw  = " ".join(from_raw)
    return sent_processed == sent_from_raw

def get_labelling(raw, processed, labels_raw, mt, bpe):
    if not(len(labels_raw) == len(raw)):
        raise Error("Number of raw tokens should match number of labels")

    from_raw = process_raw(raw, mt, bpe)

    if test_validity(processed, from_raw):
        labels_processed = extend_labels_from_raw(raw, from_raw, labels_raw)
        return labels_processed
    else:
        return None

def print_token_prediction(tokens, predictions):
    print(" ".join(["{}[{}]".format(p, t) for t, p in zip(tokens, predictions)]))

def exist_and_non_empty(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0

if __name__ == '__main__':
    data_dir = "data/xsum-detect-hall/"

    suffix = sys.argv[1]
    filename = sys.argv[2]

    if "bart" in suffix:
        mt = None
        bpe = get_encoder("data/bart/encoder.json", "data/bart/vocab.bpe")
    else:
        mt = MosesTokenizer(lang='en')
        bpe = None

    hypo_raw_filename = data_dir + "hypo_" + suffix
    hypo_processed_filename = data_dir + "hypo_processed_" + suffix
    label_raw_filename = data_dir + "label_" + suffix
    label_processed_filename = data_dir + filename

    hypo_raw = read_lines(hypo_raw_filename, str)
    hypo_processed = read_lines(hypo_processed_filename, str)
    labels_raw = read_lines(label_raw_filename, int)

    assert len(hypo_raw) == len(hypo_processed)
    assert len(hypo_raw) == len(labels_raw)

    fail_count = 0
    start = time.time()
    i = 0
    with open(label_processed_filename, 'w') as f:
        for raw, processed, l_raw in zip(hypo_raw, hypo_processed, labels_raw):
            l_processed = get_labelling(raw, processed, l_raw, mt, bpe)
            if l_processed is None:
                fail_count += 1
                result = "FAIL"
            else:
                result = " ".join(l_processed)
            print(result, file=f)

            if i % 100 == 0:
                print("i = %d" % i)

            i+=1

    end = time.time()


    fail_percent = (fail_count / len(hypo_raw)) * 100
    print("Failure percentage: %0.2f" % fail_percent)

    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))

    print("Removing files...")
    subprocess.call("rm " + hypo_raw_filename, shell=True)
    subprocess.call("rm " + hypo_processed_filename, shell=True)
    subprocess.call("rm " + label_raw_filename, shell=True)

    print("Done.")
