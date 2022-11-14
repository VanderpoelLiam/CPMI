import subprocess
import generate_lambdas
import generate_thresholds
import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Search over parameters for decoding given hallucination labels')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset model was trained on')

parser.add_argument('--score_ref', action='store_true',
                    help='Score reference sentences. By default score generated sentences.')

parser.add_argument('--runtime', type=int, default="1200",
                    help='Runtime on Euler in minutes')

args = parser.parse_args()

src_dir = "src/hallucination_labelling/"
log_dir = "logs/hallucination_labelling/"
data_dir = "xsum-hallucination"
lambdas = generate_lambdas.main()
thresholds = generate_thresholds.main()

if args.dataset == "bart":
    data_dir += "-bart"

base_params = []
if args.score_ref:
    base_params.append("-s")
    sent_type = "ref"
else:
    sent_type = "gen"

experiment_name = args.dataset
experiment_name += "_" + sent_type
log_dir += experiment_name + "/"

base_params += [args.dataset]
base_params += [data_dir]

batch_cmd = "bsub -J param_search_" + experiment_name
batch_cmd += " -o " + log_dir + "param_search"
batch_cmd += " -W " + str(args.runtime)
batch_cmd += """ \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

fairseq_cmd = "mkdir -p " + log_dir + "; "
for l in lambdas:
    for t in thresholds:
        params = base_params + [str(l)] + [str(t)]
        generate_cmd = src_dir + "generate.sh "
        generate_cmd += " ".join(params)
        fairseq_cmd += generate_cmd + "; "

final_cmd = batch_cmd + '"' + fairseq_cmd + '"'
print("Done.")
pc.copy(final_cmd)
