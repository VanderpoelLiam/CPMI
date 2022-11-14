import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Generate on test dataset with 500 labeled examples removed.')

parser.add_argument('lamb', type=float,
                    help='Lambda value')

parser.add_argument('threshold', type=float,
                    help='Entropy threshold value')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset model was trained on')

parser.add_argument('--score_ref', action='store_true',
                    help='Score reference sentences. By default score generated sentences.')

parser.add_argument('--runtime', type=int, default="200",
                    help='Runtime on Euler in minutes')

args = parser.parse_args()

src_dir = "src/"
log_dir = "logs/experiments/"
data_dir = "xsum-summarizer-no-500"

if args.dataset == "bart":
    data_dir += "-bart"

suffix = "%0.4E_%0.4E" % (args.lamb, args.threshold)
experiment_name = args.dataset

base_params = []
if args.score_ref:
    base_params.append("-s")
    experiment_name += "_ref"

experiment_name += "_no_500_" + suffix
log_dir += experiment_name
base_params += [args.dataset]
base_params += [data_dir]

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir
batch_cmd += " -W " + str(args.runtime)
batch_cmd += """ \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

params = base_params + [str(args.lamb)] + [str(args.threshold)]
generate_cmd = src_dir + "generate.sh "
generate_cmd += " ".join(params)

final_cmd = batch_cmd + '"' + generate_cmd + '"'
print("Done.")
pc.copy(final_cmd)
