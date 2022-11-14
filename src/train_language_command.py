import argparse
import pyperclip as pc
import re

parser = argparse.ArgumentParser(description='Generate fairseq-train command for language model')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset language model is trained on')

parser.add_argument('--full', action='store_true',
                    help='Use both source and targets. By default use only targets.')

parser.add_argument('--update_freq', type=int, default="32",
                    help='Frequency of parameter updates')

parser.add_argument('--lr', type=float, default="0.0005",
                    help='Learning rate')

parser.add_argument('--restore', action='store_true',
                    help='Continue training the model from the saved checkpoint')

parser.add_argument('--runtime', type=int, default="1200",
                    help='Runtime on Euler')

args = parser.parse_args()

log_dir = "logs/experiments/"
experiment_name = "train_lang"
data_source = ""
data_dir = "data/xsum-lang"
save_dir = "checkpoints/language_model/" + args.dataset

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
model_filepath = save_dir + "/checkpoint_best.pt"

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir + experiment_name
batch_cmd += " -W " + str(args.runtime)
batch_cmd += """ \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

fairseq_cmd = "fairseq-train "
fairseq_cmd += data_dir
fairseq_cmd += " --save-dir " + save_dir
fairseq_cmd += " --update-freq " + str(args.update_freq)
fairseq_cmd += " --lr " + str(args.lr)
fairseq_cmd += " --checkpoint-suffix " + suffix
if args.restore:
    fairseq_cmd += " --restore-file " + model_filepath

fairseq_cmd += """ \
--task language_modeling \
--arch transformer_lm --share-decoder-input-output-embed \
--dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
--tokens-per-sample 512 --sample-break-mode none \
--max-tokens 2048 \
--no-epoch-checkpoints --no-last-checkpoints \
--patience 5 \
"""

final_cmd = batch_cmd + fairseq_cmd

print("Done.")
pc.copy(final_cmd)
