import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Compute the token level entropy values')

parser.add_argument('--dataset', type=str, default="standard",
                    choices=["standard", "bart"],
                    help='Dataset model was trained on')

args = parser.parse_args()

data_dir = "data/xsum-hallucination"
summarization_model = "checkpoints/summarization_model/" + args.dataset + "/checkpoint_best.pt"
language_model = "checkpoints/language_model/" + args.dataset + "/checkpoint_best.pt"
experiment_name = "token_level_entropy_" + args.dataset
log_dir = "logs/experiments/" + experiment_name

if args.dataset == "bart":
    data_dir += "-bart"

fairseq_cmd = "fairseq-generate "
fairseq_cmd += data_dir
fairseq_cmd += " --path " + summarization_model
fairseq_cmd += " --lm-path " + language_model
fairseq_cmd += """ \
--batch-size 16 --beam 5 \
--score-reference \
--truncate-source \
--lm-weight 0 \
"""

batch_cmd = "bsub -J " + experiment_name
batch_cmd += " -o " + log_dir
batch_cmd += """ \
-W 200 \
-n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

final_cmd = batch_cmd + fairseq_cmd
print("Done.")
pc.copy(final_cmd)
