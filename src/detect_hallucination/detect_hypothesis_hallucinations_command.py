import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Generate hallucination labels command')

parser.add_argument('lamb', type=float,
                    help='Lambda value')

parser.add_argument('threshold', type=float,
                    help='Entropy threshold value')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset model was trained on')

parser.add_argument('--score_ref', action='store_true',
                    help='Score reference sentences. By default score generated sentences.')

parser.add_argument('--runtime', type=int, default="400",
                    help='Runtime on Euler in minutes')

args = parser.parse_args()

log_dir = "logs/detect_hallucination/"
filename = args.dataset
suffix = args.dataset

if args.score_ref:
    suffix += "_ref"
    filename += "_ref"

suffix += "_%0.4E_%0.4E" % (args.lamb, args.threshold)
filename += "_no_500_%0.4E_%0.4E" % (args.lamb, args.threshold)

experiment_name = "generate_labels_" + suffix

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir + "batch_" + suffix
batch_cmd += " -W " + str(args.runtime)
batch_cmd += """ \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

activate_detect_hall_env = "source /cluster/work/cotterell/liam/detect_hall/bin/activate; cd ../fairseq-detect-hallucination/"

activate_fair_env = "source /cluster/work/cotterell/liam/fair_env/bin/activate; cd ../master-thesis/"

src_dir = "src/detect_hallucination/"
extract_params = [filename, suffix]
if args.dataset == "bart":
    extract_params = ["-b"] + extract_params
extract_cmd = src_dir + "extract_hypos.sh " + " ".join(extract_params)

src_dir = "util_scripts/"
data_dir = " ../master-thesis/data/xsum-detect-hall/"
log_dir = " ../master-thesis/data/xsum-detect-hall/"

detect_hall_cmd = "python " + src_dir + "predict_hallucination_xsum.py "
detect_hall_cmd += "_" + suffix
detect_hall_cmd += data_dir
detect_hall_cmd += log_dir

src_dir = "src/detect_hallucination/"

filename = "label_processed_"
if args.score_ref:
    filename += args.dataset + "_ref"
else:
    filename += suffix

process_labels_cmd = "python " + src_dir + "process_labels.py " + suffix + " " + filename

cmds = []
cmds.append(extract_cmd)
cmds.append(activate_detect_hall_env)
cmds.append(detect_hall_cmd)
cmds.append(activate_fair_env)
cmds.append(process_labels_cmd)

final_cmd = batch_cmd + '"' + "; ".join(cmds) + '"'
print("Done.")
pc.copy(final_cmd)
