import argparse
import pyperclip as pc
import re

parser = argparse.ArgumentParser(description='Generate fairseq-train command for language model')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset language model is trained on')

parser.add_argument('--full', action='store_true',
                    help='Use both source and targets. By default use only targets.')

parser.add_argument('--update_freq', type=int, default="64",
                    help='Frequency of parameter updates')

parser.add_argument('--lr', type=float, default="0.0005",
                    help='Learning rate')

parser.add_argument('--wait', action='store_true',
                    help='To wait on fairseq-train to end')

args = parser.parse_args()

log_dir = "logs/experiments/"
train_name = "train_lang"
experiment_name = "eval_lang"
data_source = ""
data_dir = "data/xsum-lang"

if args.dataset == "bart":
    data_dir += "-bart"

if args.full:
    data_source += "_full"
    data_dir += "-full"

suffix = "_" + args.dataset
suffix += "_" + str(args.update_freq)
suffix += "_%0.4E" % args.lr
suffix += data_source

experiment_name += suffix
train_name += suffix
path = "checkpoints/language_model/" + args.dataset + "/checkpoint_best" + suffix + ".pt"

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir + experiment_name
if args.wait:
    batch_cmd += " -w \"ended(" + train_name + ")\""
batch_cmd += """ \
-W 60 -n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

fairseq_cmd = "fairseq-eval-lm "
fairseq_cmd += data_dir
fairseq_cmd += " --path " + path
fairseq_cmd += """ \
--batch-size 16 \
--tokens-per-sample 512 \
--context-window 400 \
"""
final_cmd = batch_cmd + fairseq_cmd

print("Done.")
pc.copy(final_cmd)
