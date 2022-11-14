import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Process generated hallucination labels command')

parser.add_argument('lamb', type=float,
                    help='Lambda value')

parser.add_argument('threshold', type=float,
                    help='Entropy threshold value')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset model was trained on')

parser.add_argument('--wait', action='store_true',
                    help='To wait on generate labels to end')

parser.add_argument('--runtime', type=int, default="240",
                    help='Runtime on Euler in minutes')

args = parser.parse_args()

log_dir = "logs/detect_hallucination/"

suffix = args.dataset + "_%0.4E_%0.4E" % (args.lamb, args.threshold)
generate_labels_name = "generate_labels_" + suffix
experiment_name = "process_labels_" + suffix

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir + "batch_" + suffix
batch_cmd += " -W " + str(args.runtime)
if args.wait:
    batch_cmd += " -w \"ended(" + generate_labels_name + ")\""
batch_cmd += """ \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

src_dir = "src/detect_hallucination/"
process_labels_cmd = "python " + src_dir + "process_labels.py " + suffix

final_cmd = batch_cmd + '"' + process_labels_cmd + '"'
print("Done.")
pc.copy(final_cmd)
