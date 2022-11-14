import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Generate gridsearch over lambda for MMI decoding command')

parser.add_argument('summarization_model', type=int,
                    help='Summarization model number')

parser.add_argument('--lang_full', action='store_true',
                    help='To use the full language model')

args = parser.parse_args()

src_dir = "src/hyperparameter_search/"

batch_cmd = "bsub -J lambda_search_%i" % args.summarization_model

if args.lang_full:
    language_model = "lang_full"
    batch_cmd += "_full"
else:
    language_model = "lang"

log_dir = "logs/hyperparameters/" + language_model + "/%i/" % args.summarization_model
batch_cmd += " -o " + log_dir + "lambda_search.log"
batch_cmd += """ \
-W 1000 -n 4 -R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

final_cmd = batch_cmd
final_cmd += "python " + src_dir + "lambda_gridsearch.py %i " % args.summarization_model
final_cmd += language_model + " "
final_cmd += log_dir

print("Done.")
pc.copy(final_cmd)
