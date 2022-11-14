
import json
import os
import subprocess
import argparse

from os.path import join
from tqdm import tqdm

def extract(in_dir, out_dir, source, target, files):
    source = open(join(out_dir, source) + ".txt", "a")
    target = open(join(out_dir, target) + ".txt", "a")

    for fname in tqdm(files):
        text_in = open(join(in_dir, fname) + ".summary").read()
        t0 = text_in.split("[SN]FIRST-SENTENCE[SN]\n", 1)[1]
        t1 = t0.split("[SN]RESTBODY[SN]\n", 1)
        source_out = t1[1].replace("\n", " ") + "\n"
        target_out = t1[0].replace("\n", " ") + "\n"
        source.write(source_out)
        target.write(target_out)

def tokenize(out_dir, split_filename, is_bart=False, is_cnn=False):
    os.system("mkdir -p " + out_dir)
    sources = ["test.source", "valid.source", "train.source"]
    targets = ["test.target", "valid.target", "train.target"]

    if not is_cnn:
        list_files = []
        split_dict = json.loads(open(split_filename).read())
        data_types = ["test", "validation", "train"]
        for type in data_types:
            list_files.append(split_dict[type])

        for source, target, files in zip(sources, targets, list_files):
            extract(in_dir, out_dir, source, target, files)

    for source, target in zip(sources, targets):
        if is_bart:
            source = join(out_dir, source)
            target = join(out_dir, target)
            os.rename(source + ".txt", source)
            os.rename(target + ".txt", target)
        else:
            subprocess.run([src_dir + 'tokenize.sh', out_dir, source, target])

def preprocess(out_dir, data_dir, lang_dir, split_filename, learn_bpe=False, is_bart=False, is_cnn=False):
    data_dir = data_dir[:-1]

    if is_bart:
        out_dir += "-bart"
    if lang_dir is not None and is_bart:
        lang_dir += "-bart"

    bpe = [src_dir + 'bpe.sh', out_dir, data_dir]
    binarize = [src_dir + 'binarize.sh', out_dir, data_dir]
    binarize_lang_model = [src_dir + 'binarize_lang_model.sh', out_dir, data_dir, lang_dir]
    learn_bpe = [src_dir + 'learn_bpe.sh', out_dir, data_dir]
    learn_dict = [src_dir + 'learn_dict.sh', out_dir, data_dir]

    if is_bart:
        bpe.insert(1, "-b")
        binarize.insert(1, "-b")
        binarize_lang_model.insert(1, "-b")

    if is_cnn:
        bpe.insert(1, "-c")
        binarize.insert(1, "-c")
        binarize_lang_model.insert(1, "-c")
        learn_bpe.insert(1, "-c")
        learn_dict.insert(1, "-c")

    # tokenize(out_dir, split_filename, is_bart, is_cnn)
    assert False
    if learn_bpe:
        subprocess.run(learn_bpe)
        subprocess.run(learn_dict)
    else:
        subprocess.run(bpe)

    subprocess.run(binarize)

    if lang_dir is not None:
        subprocess.run(binarize_lang_model)

    if 'hallucination' not in out_dir:
        subprocess.run([src_dir + 'cleanup_preprocessing.sh', out_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess XSum and CNN-Dailymail datasets')

    parser.add_argument('--all', action='store_true',
                        help='Preprocess all datasets')

    parser.add_argument('--full', action='store_true',
                        help='Preprocess the full dataset')

    parser.add_argument('--sample', action='store_true',
                        help='Preprocess small sample of the dataset')

    parser.add_argument('--labelled', action='store_true',
                        help='Preprocess 500 hallucination labelled samples of the XSum dataset')

    parser.add_argument('--no_500', action='store_true',
                        help='Preprocess the full XSum dataset without the 500 hallucination labelled samples')

    parser.add_argument('--bart', action='store_true',
                        help='Run BART specific preprocessing steps')

    parser.add_argument('--learn_bpe', action='store_true',
                        help='Learn the BPE codes from the data')

    parser.add_argument('--detect_hall', action='store_true',
                        help='Preprocess the source for fairseq-detect-hallucination')

    parser.add_argument('--cnn', action='store_true',
                        help='CNN-DM specific preprocessing steps')


    args = parser.parse_args()

    data_dir = "data/"
    src_dir = "src/preprocessing/"
    in_dir = data_dir + "bbc-summary-data"
    dataset = "xsum-"

    if args.cnn:
        dataset = "cnn-dm-"
        if not (args.sample or args.full):
            raise NotImplementedError()

    if args.learn_bpe and not args.full:
        raise Error("Can only learn bpe for full model")

    if args.full or args.all:
        # TODO: Need to be careful now that I will have different codes, dict.txt and vocab.bpe for Xsum, CNN-DailyMail and BART

        out_dir = data_dir + dataset + "summarizer"
        lang_dir = data_dir + dataset + "lang"
        split_filename = data_dir + "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
        preprocess(out_dir, data_dir, lang_dir, split_filename, is_bart=args.bart, learn_bpe=args.learn_bpe, is_cnn=args.cnn)

    if args.sample or args.all:
        out_dir = data_dir + dataset + "summarizer-samples"
        lang_dir = data_dir + dataset + "lang-samples"
        split_filename = data_dir + "XSum-samples-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
        preprocess(out_dir, data_dir, lang_dir, split_filename, is_bart=args.bart, is_cnn=args.cnn)

    if args.labelled or args.all:
        out_dir = data_dir + "xsum-hallucination"
        split_filename = data_dir + "XSum-hallucination-split.json"
        preprocess(out_dir, data_dir, None, split_filename, is_bart=args.bart)

    if args.no_500 or args.all:
        out_dir = data_dir + "xsum-summarizer-no-500"
        split_filename = data_dir + "XSum-no-500-split.json"
        preprocess(out_dir, data_dir, None, split_filename, is_bart=args.bart)

    if args.detect_hall or args.all:
        out_dir = data_dir + "xsum-detect-hall"
        split_filename = data_dir + "XSum-no-500-split.json"
        os.system("mkdir -p " + out_dir)

        split_dict = json.loads(open(split_filename).read())
        extract(in_dir, out_dir, "test.source", "test.target", split_dict["test"])
        subprocess.call("rm " + out_dir + "/test.target.txt", shell=True)
        subprocess.run([src_dir + 'norm_text.sh', out_dir, "test.source"])
